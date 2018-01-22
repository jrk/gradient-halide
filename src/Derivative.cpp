#include "Derivative.h"

#include "DerivativeUtils.h"
#include "BoundaryConditions.h"
#include "Simplify.h"
#include "Substitute.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IREquality.h"
#include "FindCalls.h"
#include "RealizationOrder.h"
#include "Error.h"

#include <iostream>
#include <cmath>

namespace Halide {
namespace Internal {

/** An IR visitor that computes the derivatives through reverse accumulation
 */
class ReverseAccumulationVisitor : public IRVisitor {
public:
    using IRVisitor::visit;

    void propagate_adjoints(const Func &output,
                            const Func &adjoint,
                            const std::vector<std::pair<Expr, Expr>> &output_bounds);

    std::map<FuncKey, Func> get_adjoint_funcs() const {
        return adjoint_funcs;
    }

protected:
    void visit(const Cast *op);
    void visit(const Variable *op);
    void visit(const Add *op);
    void visit(const Sub *op);
    void visit(const Mul *op);
    void visit(const Div *op);
    void visit(const Min *op);
    void visit(const Max *op);
    void visit(const Let *op);
    void visit(const Select *op);
    void visit(const Call *op);

private:
    void accumulate(const Expr &stub, const Expr &adjoint);

    std::map<const BaseExprNode *, Expr> expr_adjoints;
    std::map<FuncKey, Func> adjoint_funcs;
    std::map<std::string, Expr> let_var_mapping;
    std::vector<std::string> let_variables;
    std::map<std::string, Box> func_bounds;
    Func current_func;
    int current_update_id;
};

void ReverseAccumulationVisitor::propagate_adjoints(
        const Func &output,
        const Func &adjoint,
        const std::vector<std::pair<Expr, Expr>> &output_bounds) {
    // Topologically sort the functions
    std::map<std::string, Function> env = find_transitive_calls(output.function());
    std::vector<std::string> order = realization_order({output.function()}, env).first;
    std::vector<Func> funcs;
    funcs.reserve(order.size());
    // Internal::debug(0) << "Sorted Func list:" << "\n";
    // for (const auto &func_name : order) {
    //     Internal::debug(0) << "  . " << func_name << "\n";
    // }
    for (const auto &func_name : order) {
        Func func(env[func_name]);
        funcs.push_back(Func(env[func_name]));
    }

    internal_assert(funcs.size() > 0);

    // debug(0) << "ReverseAccumulationVisitor: infering bounds...";
    func_bounds = inference_bounds(output, output_bounds);
    // debug(0) << "done\n";

    // Create a stub for each function to accumulate adjoints.
    for (int func_id = 0; func_id < (int)funcs.size(); func_id++) {
        const Func &func = funcs[func_id];
        for (int update_id = -1; update_id < func.num_update_definitions(); update_id++) {
            Func adjoint_func(func.name() + "_" + std::to_string(update_id + 1) + "_d_def__");
            bool is_final_output = func_id == (int)funcs.size() - 1 &&
                                   update_id == func.num_update_definitions() - 1;
            std::vector<Var> args = func.args();
            for (auto &arg : args) {
                if (arg.is_implicit()) {
                    // replace implicit variables with non implicit ones
                    arg = Var();
                }
            }
            if (is_final_output) {
                adjoint_func(args) = adjoint(args);
            } else {
                if (func.values().size() == 1) {
                    adjoint_func(args) = make_zero(func.output_types()[0]);
                } else {
                    std::vector<Expr> init(func.values().size());
                    for (size_t i = 0; i < func.values().size(); i++) {
                        init[i] = make_zero(func.output_types()[i]);
                    }
                    adjoint_func(args) = Tuple(init);
                }
            }
            FuncKey func_key{func.name(), update_id};
            assert(adjoint_funcs.find(func_key) == adjoint_funcs.end());
            adjoint_funcs[func_key] = adjoint_func;
        }
    }
    // Also create stubs for buffers
    std::map<std::string, int> buffers_dimensions;
    for (int func_id = 0; func_id < (int)funcs.size(); func_id++) {
        const Func &func = funcs[func_id];
        std::map<std::string, int> func_buffers_dimensions = find_buffers_dimensions(func);
        buffers_dimensions.insert(func_buffers_dimensions.begin(), func_buffers_dimensions.end());
    }
    for (const auto &it : buffers_dimensions) {
        Func adjoint_func(it.first + "_d__");
        std::vector<Var> args;
        for (int i = 0; i < it.second; i++) {
            args.push_back(Var());
        }
        adjoint_func(args) = Expr(0.0); // TODO: pick the right type
        FuncKey func_key{it.first, -1};
        if (adjoint_funcs.find(func_key) != adjoint_funcs.end()) {
            user_error << "Naming conflict between buffer and function:" << it.first << "\n";
        }
        adjoint_funcs[func_key] = adjoint_func;
    }

    // Traverse functions from producers to consumers for reverse accumulation
    for (int func_id = funcs.size() - 1; func_id >=  0; func_id--) {
        const Func &func = funcs[func_id];
        current_func = func;

        // Traverse from the last update to first
        for (int update_id = func.num_update_definitions() - 1;
                update_id >= -1; update_id--) {
            current_update_id = update_id;
            FuncKey func_key{func.name(), update_id};
            internal_assert(func_bounds.find(func.name()) != func_bounds.end());

            // Set up boundary condition if this is the first visit to the function
            if (update_id == func.num_update_definitions() - 1 &&
                    func.dimensions() > 0) {
                Func &adjoint_func = adjoint_funcs[func_key];

                const Box &bounds = func_bounds[func.name()];

                // Save a pointer to the unbounded def. Useful for scheduling
                FuncKey unbounded_func_key{func.name() + "_unbounded", update_id};
                adjoint_funcs[unbounded_func_key] = adjoint_func;

                if (adjoint_func.values().size() == 1) {
                    adjoint_func = BoundaryConditions::constant_exterior(
                            adjoint_func, make_zero(adjoint_func.output_types()[0]), box_to_vector(bounds),
                            adjoint_func.name() + "_ce");
                } else {
                    std::vector<Expr> values(adjoint_func.values().size());
                    for (size_t i = 0; i < values.size(); i++) {
                        values[i] = make_zero(adjoint_func.output_types()[i]);
                    }
                    adjoint_func = BoundaryConditions::constant_exterior(
                            adjoint_func, Tuple(values), box_to_vector(bounds),
                            adjoint_func.name() + "_ce");
                }
            }

            // Initialize the next adjoint function by propagating the adjoints to next update
            // Example:
            // f(x) = ...
            // f(1) = ... <- we're here
            // We have an adjoint for f(1) defined over the whole support of f
            // Now we want to initialize for the f(x) update
            // Need to propagate back to all x while masking 1
            // x -> next_args
            // 1 -> update_args
            if (update_id >= 0) {
                FuncKey next_func_key{func.name(), update_id - 1};
                Func &next_adjoint_func = adjoint_funcs[next_func_key];
                std::vector<Var> next_args = next_adjoint_func.args();
                std::vector<Expr> update_args = func.update_args(update_id);
                // Replace implicit variables
                for (auto &arg : update_args) {
                    std::set<std::string> implicit_variables = find_implicit_variables(arg);
                    for (const auto &var : implicit_variables) {
                        arg = substitute(var, next_args[Var::implicit_index(var)], arg);
                    }
                }
                // Check if next_args are the same as update_args
                // If they are the same simply set everything to zero
                bool is_noop = true;
                for (int i = 0 ; i < (int)next_args.size(); i++) {
                    const Variable *update_var = update_args[i].as<Variable>();
                    if (update_var == nullptr || next_args[i].name() != update_var->name) {
                        is_noop = false;
                    }
                }
                next_adjoint_func = Func(next_adjoint_func.name());
                if (!is_noop) {
                    // f'(x) = adjoint
                    next_adjoint_func(next_args) =
                        adjoint_funcs[func_key](next_args);
                }
                if (func.values().size() == 1) {
                    next_adjoint_func(update_args) = Expr(0.0); // TODO: pick the right type
                } else {
                    std::vector<Expr> init(func.values().size(), Expr(0.0)); // TODO: pick the right type
                    next_adjoint_func(update_args) = Tuple(init);
                }
            }

            // Now we want to propagate the derivatives at expression level
            // Topologically sort the expressions for each value in the tuple
            std::vector<Expr> expr_list;
            Tuple tuple = update_id < 0 ? func.values() : func.update_values(update_id);
            std::vector<const BaseExprNode *> output_exprs;
            auto tuple_vector = tuple.as_vector();
            for (const auto &expr : tuple_vector) {
                std::vector<Expr> value_expr_list = sort_expressions(expr);
                expr_list.insert(expr_list.end(), value_expr_list.begin(), value_expr_list.end());
                output_exprs.push_back((const BaseExprNode *)expr_list.back().get());
            }

            // TODO: replace let_var_mapping with Scope
            // Gather let variables
            let_var_mapping.clear();
            let_variables.clear();
            for (auto it = expr_list.begin(); it != expr_list.end(); it++) {
                Expr expr = *it;
                if (expr.get()->node_type == IRNodeType::Let) {
                    const Let *op = expr.as<Let>();
                    // Assume Let variables are unique
                    assert(let_var_mapping.find(op->name) == let_var_mapping.end());
                    let_var_mapping[op->name] = op->value;
                    let_variables.push_back(op->name);
                }
            }

            // Retrieve previously propagated adjoint for the Func,
            // apply it to expression adjoints
            std::vector<Expr> update_args;
            if (update_id >= 0) {
                update_args = func.update_args(update_id);
            } else {
                update_args.reserve(func.args().size());
                Func adjoint_func = adjoint_funcs[func_key];
                for (const auto &var : adjoint_func.args()) {
                    update_args.push_back(var);
                }
            }
            for (int i = 0; i < (int)output_exprs.size(); i++) {
                expr_adjoints[output_exprs[i]] =
                    Call::make(adjoint_funcs[func_key].function(), update_args, i);
            }

            // Traverse the expressions in reverse order
            for (auto it = expr_list.rbegin(); it != expr_list.rend(); it++) {
                // Propagate adjoints
                it->accept(this);
            }
            expr_adjoints.clear();
        }
    }
}

void ReverseAccumulationVisitor::accumulate(const Expr &stub, const Expr &adjoint) {
    const BaseExprNode *stub_ptr = (const BaseExprNode *)stub.get();
    if (expr_adjoints.find(stub_ptr) == expr_adjoints.end()) {
        expr_adjoints[stub_ptr] = adjoint;
    } else {
        expr_adjoints[stub_ptr] += adjoint;
    }
}

void ReverseAccumulationVisitor::visit(const Cast *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/dx cast(x) = 1
    accumulate(op->value, adjoint);
}

void ReverseAccumulationVisitor::visit(const Variable *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    auto it = let_var_mapping.find(op->name);
    if (it != let_var_mapping.end()) {
        accumulate(it->second, Let::make(op->name, it->second, adjoint));
    }
}

void ReverseAccumulationVisitor::visit(const Add *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/da a + b = 1
    accumulate(op->a, adjoint);
    // d/db a + b = 1
    accumulate(op->b, adjoint);
}

void ReverseAccumulationVisitor::visit(const Sub *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/da a - b = 1
    accumulate(op->a, adjoint);
    // d/db a - b = -1
    accumulate(op->b, -adjoint);
}

void ReverseAccumulationVisitor::visit(const Mul *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/da a * b = b
    accumulate(op->a, adjoint * op->b);
    // d/db a * b = a
    accumulate(op->b, adjoint * op->a);
}

void ReverseAccumulationVisitor::visit(const Div *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/da a / b = 1 / b
    accumulate(op->a, adjoint / op->b);
    // d/db a / b = - a / b^2
    accumulate(op->b, - adjoint * op->a / (op->b * op->b));
}

void ReverseAccumulationVisitor::visit(const Min *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/da min(a, b) = a <= b ? 1 : 0
    accumulate(op->a, select(op->a <= op->b, adjoint, make_zero(op->type)));
    // d/db min(a, b) = b <= a ? 1 : 0
    accumulate(op->b, select(op->b <= op->a, adjoint, make_zero(op->type)));
}

void ReverseAccumulationVisitor::visit(const Max *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/da max(a, b) = a >= b ? 1 : 0
    accumulate(op->a, select(op->a >= op->b, adjoint, make_zero(op->type)));
    // d/db max(a, b) = b >= a ? 1 : 0
    accumulate(op->b, select(op->b >= op->a, adjoint, make_zero(op->type)));
}

void ReverseAccumulationVisitor::visit(const Let *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    accumulate(op->body, adjoint);
}

void ReverseAccumulationVisitor::visit(const Select *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/db select(a, b, c) = select(a, 1, 0)
    accumulate(op->true_value, select(op->condition, adjoint, make_zero(op->type)));
    // d/dc select(a, b, c) = select(a, 0, 1)
    accumulate(op->false_value, select(op->condition, make_zero(op->type), adjoint));
}

void ReverseAccumulationVisitor::visit(const Call *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];
    // Math functions
    if (op->is_extern()) {
      if (op->name == "exp_f32") {
          // d/dx exp(x) = exp(x)
          accumulate(op->args[0], adjoint * exp(op->args[0]));
      } else if (op->name == "log_f32" || op->name == "log_f64") {
          // d/dx log(x) = 1 / x
          accumulate(op->args[0], adjoint / op->args[0]);
      } else if (op->name == "sin_f32" || op->name == "sin_f64") {
          // d/dx sin(x) = cos(x)
          accumulate(op->args[0], adjoint * cos(op->args[0]));
      } else if (op->name == "asin_f32" || op->name == "asin_f64") {
          // d/dx asin(x) = 1 / sqrt(1 - x^2)
          accumulate(op->args[0], adjoint / sqrt(1 - pow(op->args[0], 2)));
      } else if (op->name == "cos_f32" || op->name == "cos_f64") {
          // d/dx cos(x) = -sin(x)
          accumulate(op->args[0], - adjoint * sin(op->args[0]));
      } else if (op->name == "ceil_f32") {
          // TODO: d/dx = dirac(n) for n in Z ...
          accumulate(op->args[0], 0.0f);
      } else if (op->name == "floor_f32") {
          // TODO: d/dx = dirac(n) for n in Z ...
          accumulate(op->args[0], 0.0f);
      } else if (op->name == "sqrt_f32") {
          accumulate(op->args[0], adjoint * 0.5f / sqrt(op->args[0]));
      } else if (op->name == "sqrt_f64") {
          accumulate(op->args[0], adjoint * Expr(0.5) / sqrt(op->args[0]));
      } else if (op->name == "pow_f32") {
          accumulate(op->args[0], adjoint * op->args[1] * pow(op->args[0], op->args[1] - 1.f));
          accumulate(op->args[1], adjoint * pow(op->args[0], op->args[1]) * log(op->args[0]));
      } else if (op->name == "halide_print") {
          accumulate(op->args[0], make_zero(op->args[0].type()));
      } else {
          internal_error << "The derivative of " << op->name << " is not implemented.";
      }
    } else if (op->call_type == Call::Halide ||
               op->call_type == Call::Image) { // Halide function call or Halid buffer
        // TODO: check if we need this elsewhere
        // Add Let expressions
        adjoint = add_let_expression(adjoint, let_var_mapping, let_variables);
        std::vector<Expr> lhs = op->args;
        for (int i = 0; i < (int)lhs.size(); i++) {
            lhs[i] = add_let_expression(lhs[i], let_var_mapping, let_variables);
        }

        // If target is the current function itself, send to previous update
        // e.g. f(x) = ...
        //      f(x) = f(x) + 1
        FuncKey func_key;
        if (op->func.defined()) {
            Function func(op->func);
            func_key = func.name() != current_func.name() ?
                       FuncKey{func.name(), func.updates().size() - 1} :
                       FuncKey{func.name(), current_update_id - 1};
        } else {
            func_key = FuncKey{op->name, -1};
        }
        assert(adjoint_funcs.find(func_key) != adjoint_funcs.end());
        Func& func_to_update = adjoint_funcs[func_key];
        assert(func_to_update.dimensions() == (int)lhs.size());

        bool debug_flag = false;//func_key.first == "f_grid";

        if (debug_flag) {
            debug(0) << "current_func:" << current_func.name() << "\n";
            debug(0) << "Scattering to " << op->name << "\n";
            debug(0) << "lhs is:";
            for (const auto &arg : lhs) {
                debug(0) << " " << arg;
            }
            debug(0) << "\n";
            debug(0) << "adjoint is:" << simplify(adjoint) << "\n";
            //PrintFuncOptions options;
            //options.depth = 1;
            //print_func(current_func, options);
        }

        // Gather argument & bounds information
        // current_args are the pure variables
        // current_update_args are the actual updates at left hand side
        Func current_adjoint_func =
            adjoint_funcs[FuncKey{current_func.name(), current_update_id}];
        std::vector<Var> current_args = current_adjoint_func.args();
        std::vector<Expr> current_update_args;
        if (current_update_id >= 0) {
            current_update_args = current_func.update_args(current_update_id);
        } else {
            current_update_args.reserve(current_args.size());
            for (const auto &var : current_args) {
                current_update_args.push_back(var);
            }
        }
        const Box &current_bounds = func_bounds[current_func.name()];

        // Replace implicit variables
        for (auto &arg : lhs) {
            std::set<std::string> implicit_variables = find_implicit_variables(arg);
            for (const auto &var : implicit_variables) {
                arg = substitute(var, current_args[Var::implicit_index(var)], arg);
            }
        }
        {
            std::set<std::string> implicit_variables = find_implicit_variables(adjoint);
            for (const auto &var : implicit_variables) {
                adjoint = substitute(var, current_args[Var::implicit_index(var)], adjoint);
            }
        }

        // We want to do this:
        // func_to_update(op->args) += adjoint(current_update_args);
        // But op->args can be invalid lhs, need to canonicalize
        // we canonicalize by first trying to substitute with pure variables
        // if that fails we will replace variables on lhs with RDoms

        // TODO: maybe modularize this more
        // Try to do the canonicalization by substitution of variables
        // Create a set of new substitution variables
        std::vector<Var> new_args;
        std::set<std::string> new_args_set;
        for (int i = 0; i < (int)func_to_update.args().size(); i++) {
            new_args.push_back(Var("u" + std::to_string(i) + "_"));
            new_args_set.insert(new_args.back().name());
        }

        // We canonicalize the left hand side arguments (op->args) so that it's always x, y, z, ...
        //
        // Given:
        // g(x, y, z) = f(x, y-1, z+1)
        // we get:
        // d_f(x, y - 1, z + 1) += d_g(x,y,z)
        // Goal: rewrite to
        //  ==> d_f(x,y,z) += d_g(x,y+1,z-1)
        //
        // First gather all arguments that contains multiple pure variables
        // we don't want to mess with system solving yet, so we invalidate all of them
        // This is currently simple and conservative, solving each dimension independently, so
        // inter-dependencies like:
        // g(x, y) = f(x*y, x+y)
        // can't be simplified. In principle this can be inverted by solving a system of equations.
        // (TODO: make this a routine? also the invalidated vars?)
        std::vector<bool> canonicalized(lhs.size(), false);
        std::set<std::string> canonicalized_vars;
        std::map<std::string, Var> lhs_substitute_map;
        for (int i = 0; i < (int)lhs.size(); i++) {
            // Gather all pure variables at op->args[i], substitute them with new_args
            // For now only support single pure variable
            std::vector<std::string> variables =
                gather_variables(lhs[i], vars_to_strings(current_args));
            if (variables.size() != 1) {
                continue;
            }

            bool solved;
            Expr result_rhs;
            std::tie(solved, result_rhs) =
                solve_inverse(new_args[i] == lhs[i],
                              new_args[i].name(),
                              variables[0]);
            if (!solved) {
                continue;
            }
            // debug(0) << "result : " << result_rhs << "\n";

            // Replace pure variable with the reverse
            adjoint = substitute(variables[0], result_rhs, adjoint);

            lhs[i] = func_to_update.args()[i];
            canonicalized[i] = true;
            canonicalized_vars.insert(variables[0]);
            lhs_substitute_map[variables[0]] = func_to_update.args()[i];
        }

        // Deal with the case where the two functions have different set of variables
        for (int i = 0; i < (int)lhs.size(); i++) {
            for (const auto &it : lhs_substitute_map) {
                lhs[i] = substitute(it.first, it.second, lhs[i]);
            }
        }

        // Sometimes the canonicalization above doesn't work.
        // We replace the pure variables inside lhs with RDoms for general scattering
        // First find the corresponding current argument to obtain the bound
        std::vector<std::pair<Expr, Expr>> bounds;
        bounds.reserve(current_args.size());
        for (int arg_id = 0; arg_id < (int)current_args.size(); arg_id++) {
            bounds.push_back({current_bounds[arg_id].min,
                              current_bounds[arg_id].max - current_bounds[arg_id].min + 1});
        }
        RDom r_bounds(bounds);
        for (int lhs_id = 0; lhs_id < (int)lhs.size(); lhs_id++) {
            Expr lhs_arg = lhs[lhs_id];
            if (!canonicalized[lhs_id]) {
                std::vector<std::string> variables =
                    gather_variables(lhs_arg, current_adjoint_func.function().args());
                RDom r(bounds);
                for (int var_id = 0; var_id < (int)variables.size(); var_id++) {
                    for (int arg_id = 0; arg_id < (int)current_args.size(); arg_id++) {
                        if (current_args[arg_id].name() == variables[var_id] &&
                                canonicalized_vars.find(current_args[arg_id].name()) ==
                                    canonicalized_vars.end()) {
                            lhs[lhs_id] = substitute(variables[var_id],
                                                     r_bounds[arg_id],
                                                     lhs[lhs_id]);
                            adjoint = substitute(variables[var_id], r_bounds[arg_id], adjoint);
                            break;
                        }
                    }
                }
            }
        }

        // For each free variable on the rhs, replace it with current bounds
        // e.g. we have in forward pass f(x, y) = g(x)
        //      then we need to do g'(x) += f(x, y)
        //      now we need to replace y with a reduction variable over f's bound
        //      x is automatically excluded since it's
        //      replaced by the new substitution variable e.g. u_0

        // First gather all free variables
        FuncBounds bounds_subset;
        std::vector<int> arg_id_to_substitute;
        bounds_subset.reserve(current_args.size());
        arg_id_to_substitute.reserve(current_args.size());
        for (int i = 0; i < (int)current_args.size(); i++) {
            if (has_variable(adjoint, current_args[i].name())) {
                const Interval &interval = current_bounds[i];
                bounds_subset.emplace_back(interval.min, interval.max - interval.min + 1);
                arg_id_to_substitute.push_back(i);
            }
        }

        // Create a new RDom to loop over all free variables
        if (arg_id_to_substitute.size() > 0) {
            RDom r(bounds_subset);
            for (int i = 0; i < (int) arg_id_to_substitute.size(); i++) {
                int arg_id = arg_id_to_substitute[i];
                adjoint = substitute(current_args[arg_id].name(), r[i], adjoint);
            }
        }

        // General scattering simplification rules
        // For each expression in lhs, check if it is an expression of a single rvar and
        // spans the same interval of the function's bound
        // if so we can rewrite it back to pure variables
        // e.g.
        // f(r.x) = g(r.x)
        // => f(x) = g(x)
        // Another common pattern is the reverse of downsampling
        // if we see s * r.x + r.y and r.y has min == 0 and extent == s
        // we simplify them to x and replace all occurence of r.x by x/4
        // e.g.
        // f(4 * r.x + r.y) = g(r.x) + h(4 * r.x + r.y)
        // => f(x) = g(x/4) + h(x)
        std::vector<Var> func_to_update_args = func_to_update.args();
        for (int i = 0; i < (int)lhs.size(); i++) {
            Expr lhs_arg = substitute_in_all_lets(lhs[i]);
            const Variable *var = lhs_arg.as<Variable>();
            const Add *add = lhs_arg.as<Add>();
            if (var != nullptr && var->reduction_domain.defined()) {
                ReductionDomain rdom = var->reduction_domain;
                int rvar_id = -1;
                for (int rid = 0; rid < (int)rdom.domain().size(); rid++) {
                    if (rdom.domain()[rid].var == var->name) {
                        rvar_id = rid;
                        break;
                    }
                }
                assert(rvar_id != -1);
                ReductionVariable rvar = rdom.domain()[rvar_id];
                // Check if the min/max of the rvariable is the same as the target function
                const Box &target_bounds = func_bounds[op->name];
                Interval t_interval = target_bounds[i];
                t_interval.min = simplify(t_interval.min);
                t_interval.max = simplify(t_interval.max);
                Interval r_interval(simplify(rvar.min),
                                    simplify(rvar.min + rvar.extent - 1));
                if (can_prove(r_interval.min <= t_interval.min &&
                              r_interval.max >= t_interval.max)) {
                    lhs[i] = func_to_update_args[i];
                    // Replace other occurence of rvar in lhs
                    for (int j = 0; j < (int)lhs.size(); j++) {
                        if (j != i) {
                            lhs[j] = simplify(substitute(rvar.var, func_to_update_args[i], lhs[j]));
                        }
                    }
                    adjoint = simplify(substitute(rvar.var, func_to_update_args[i], adjoint));
                }
            } else if (add != nullptr &&
                       ((add->a.as<Mul>() != nullptr && add->b.as<Variable>() != nullptr) ||
                        (add->a.as<Variable>() != nullptr && add->b.as<Mul>() != nullptr))) {
                // Find pattern s * r.x + r.y where r.y.min == 0 && r.y.extent == s
                Expr a = add->a, b = add->b;
                if (add->b.as<Mul>() != nullptr) {
                    // swap so that b is always the Variable
                    assert(add->a.as<Variable>() != nullptr);
                    std::swap(a, b);
                }
                const Mul *mul = a.as<Mul>();
                const Variable *b_var = b.as<Variable>();
                assert(mul != nullptr && b_var != nullptr);
                Expr mul_a = mul->a, mul_b = mul->b;
                if (mul_a.as<Variable>() != nullptr &&
                        mul_a.as<Variable>()->reduction_domain.defined()) {
                    std::swap(mul_a, mul_b);
                }
                const Variable *mul_b_var = mul_b.as<Variable>();
                if (mul_b_var == nullptr || !mul_b_var->reduction_domain.defined()) {
                    continue;
                }
                ReductionDomain b_rdom = b_var->reduction_domain;
                if (!b_rdom.defined()) {
                    continue;
                }

                int rvar_id = -1;
                for (int rid = 0; rid < (int)b_rdom.domain().size(); rid++) {
                    if (b_rdom.domain()[rid].var == b_var->name) {
                        rvar_id = rid;
                        break;
                    }
                }
                assert(rvar_id != -1);
                ReductionVariable rvar = b_rdom.domain()[rvar_id];
                if (!equal(rvar.min, Expr(0)) || !equal(rvar.extent, mul_a)) {
                    continue;
                }

                // We've finally made sure that the expression has the form we want
                // Now replace everything
                // replace s * r.x + r.y with x
                lhs[i] = func_to_update_args[i];
                adjoint = substitute(lhs_arg,
                                     func_to_update_args[i],
                                     substitute_in_all_lets(adjoint));
                // replace r.x with x / s
                adjoint = substitute(mul_b, func_to_update_args[i] / mul_a, adjoint);
                adjoint = simplify(adjoint);
            }
        }

        // TODO: can we somehow enforce the same partial ordering?
        // Merge RDoms on both lhs and rhs
        std::map<std::string, std::pair<Expr, Expr>> rvar_maps =
            gather_rvariables(adjoint);
        for (const auto &lhs_arg : lhs) {
            std::map<std::string, std::pair<Expr, Expr>> maps =
                gather_rvariables(lhs_arg);
            rvar_maps.insert(maps.begin(), maps.end());
        }
        std::vector<std::string> var_names;
        FuncBounds merged_bounds;
        for (const auto &it : rvar_maps) {
            var_names.push_back(it.first);
            merged_bounds.emplace_back(it.second.first, it.second.second);
        }
        RDom merged_r;
        if (merged_bounds.size() > 0) {
            merged_r = RDom(merged_bounds);
            for (int i = 0; i < merged_r.dimensions(); i++) {
                adjoint = substitute(var_names[i], merged_r[i], adjoint);
                for (auto &lhs_arg : lhs) {
                    lhs_arg = substitute(var_names[i], merged_r[i], lhs_arg);
                }
            }
        }

        for (int i = 0; i < (int)func_to_update_args.size(); i++) {
            // substitute the new_args back to original variables
            adjoint = substitute(new_args[i].name(), func_to_update_args[i], adjoint);
        }
        adjoint = simplify(adjoint);

        // Finally we update the function definitions, possibly merge with previous updates
        auto can_merge = [&]() -> bool {
            if (func_to_update.num_update_definitions() == 0) {
                // If lhs are not pure variables we can't merge to pure definition
                for (int i = 0; i < (int)lhs.size(); i++) {
                    if (!equal(lhs[i], func_to_update.args()[i])) {
                        return false;
                    }
                }
                ReductionDomain rdom = extract_rdom(adjoint);
                // If there are rdoms in adjoint we can't merge
                return !rdom.defined();
            }
            int update_id = func_to_update.num_update_definitions() - 1;
            std::vector<Expr> prev_lhs =
                func_to_update.update_args(update_id);
            assert(prev_lhs.size() == lhs.size());
            // If previous update has different left hand side, can't merge
            for (int i = 0; i < (int)prev_lhs.size(); i++) {
                if (!equal(lhs[i], prev_lhs[i])) {
                    return false;
                }
            }
            // If previous update has a different set of reduction variables, can't merge
            const std::vector<ReductionVariable> &rvars =
                func_to_update.update(update_id).get_schedule().rvars();
            if (!merged_r.defined()) {
                return rvars.size() == 0;
            }
            if ((int)rvars.size() != merged_r.dimensions()) {
                return false;
            }

            for (int i = 0; i < (int)rvars.size(); i++) {
                if (!equal(rvars[i].min, merged_r[i].min())) {
                    return false;
                }
                if (!equal(rvars[i].extent, merged_r[i].extent())) {
                    return false;
                }
            }
            return true;
        };

        if (debug_flag) {
            debug(0) << "func_to_update.name():" << func_to_update.name() << "\n";
            debug(0) << "lhs after canonicalization:";
            for (const auto &arg : lhs) {
                debug(0) << " " << arg;
            }
            debug(0) << "\n";
            debug(0) << "adjoint after canonicalization:" << simplify(adjoint) << "\n";
        }

        // TODO: maybe do some analysis on lhs to avoid applying boundary conditions to
        //       function calls in adjoint
        if (!can_merge()) {
            if (func_to_update.values().size() == 1) {
                func_to_update(lhs) += adjoint;
            } else {
                func_to_update(lhs)[op->value_index] += adjoint;
            }
        } else {
            Definition &def = func_to_update.num_update_definitions() == 0 ?
                func_to_update.function().definition() :
                func_to_update.function().update(
                        func_to_update.num_update_definitions() - 1);
            std::vector<Expr> &values = def.values();
            ReductionDomain rdom;
            for (const auto &val : values) {
                rdom = extract_rdom(val);
                if (rdom.defined()) {
                    break;
                }
            }
            if (rdom.defined()) {
                assert(func_to_update.num_update_definitions() > 0);
                // Make sure we're using the same set of reduction variables
                for (int i = 0; i < merged_r.dimensions(); i++) {
                    adjoint = substitute(merged_r[i].name(), RVar(rdom, i), adjoint);
                }
            }

            if (values.size() == 1) {
                values[0] = simplify(values[0] + adjoint);
            } else {
                const Add *add = values[op->value_index].as<Add>();
                if (add != nullptr &&
                        add->b.as<Call>() != nullptr &&
                        add->b.as<Call>()->is_intrinsic(Call::undef)) {
                    // Sometimes the expression is an undef for the case of a tuple.
                    // Make sure we don't include the undefs
                    values[op->value_index] = simplify(add->a + adjoint);
                } else {
                    values[op->value_index] = simplify(values[op->value_index] + adjoint);
                }
            }
        }

        if (debug_flag) {
            //print_func(func_to_update);
        }
    } else {
        internal_assert(op->is_intrinsic());
        if (op->is_intrinsic(Call::abs)) {
            accumulate(op->args[0], adjoint*select(op->args[0] > 0, make_one(op->args[0].type()), -make_one(op->args[0].type())));
        } else if (op->is_intrinsic(Call::lerp)) {
            // z = x * (1 - w) + y * w
            // dz/dx = 1 - w
            // dz/dy = w
            // dz/dw = y - x
            accumulate(op->args[0], adjoint * (1 - op->args[2]));
            accumulate(op->args[1], adjoint * op->args[2]);
            accumulate(op->args[2], adjoint * (op->args[1] - op->args[0]));
        } else if (op->is_intrinsic(Call::likely)) {
            accumulate(op->args[0], adjoint);
        } else if (op->is_intrinsic(Call::return_second)) {
            accumulate(op->args[0], make_zero(op->args[0].type()));
            accumulate(op->args[1], adjoint);
        } else if (op->is_intrinsic(Call::undef)) {
            // do nothing
        } else {
            user_warning << "Dropping gradients at call to " << op->name << "\n";
            for (const auto &arg : op->args) {
                accumulate(arg, make_zero(arg.type()));
            }
        }
    }
}

Expr forward_accumulation(const Expr &expr,
                          const std::map<std::string, Func> &tangents,
                          Scope<Expr> &scope) {
    if (const Cast *op = expr.as<Cast>()) {
        Expr t = forward_accumulation(op->value, tangents, scope);
        return Cast::make(op->type, t);
    } else if (const Add *op = expr.as<Add>()) {
        // d/dx f(x) + g(x) = d/dx f(x) + d/dx g(x)
        Expr a = forward_accumulation(op->a, tangents, scope);
        Expr b = forward_accumulation(op->b, tangents, scope);
        return a + b;
    } else if (const Sub *op = expr.as<Sub>()) {
        // d/dx f(x) - g(x) = d/dx f(x) - d/dx g(x)
        Expr a = forward_accumulation(op->a, tangents, scope);
        Expr b = forward_accumulation(op->b, tangents, scope);
        return a - b;
    } else if (const Mul *op = expr.as<Mul>()) {
        // d/dx f(x) g(x) = g(x) d/dx f(x) + f(x) d/dx g(x)
        Expr a = forward_accumulation(op->a, tangents, scope);
        Expr b = forward_accumulation(op->b, tangents, scope);
        return simplify(op->a * b + a * op->b);
    } else if (const Div *op = expr.as<Div>()) {
        // d/dx f(x) / g(x) = (f'g - g'f) / g^2
        Expr a = forward_accumulation(op->a, tangents, scope);
        Expr b = forward_accumulation(op->b, tangents, scope);
        return simplify(((op->b * a - op->a * b) / (op->b * op->b)));
    } else if (const Min *op = expr.as<Min>()) {
        Expr a = forward_accumulation(op->a, tangents, scope);
        Expr b = forward_accumulation(op->b, tangents, scope);
        return simplify(select(op->a < op->b, a, b));
    } else if (const Max *op = expr.as<Max>()) {
        Expr a = forward_accumulation(op->a, tangents, scope);
        Expr b = forward_accumulation(op->b, tangents, scope);
        return simplify(select(op->a > op->b, a, b));
    } else if (const Select *op = expr.as<Select>()) {
        Expr true_value = forward_accumulation(op->true_value, tangents, scope);
        Expr false_value = forward_accumulation(op->false_value, tangents, scope);
        return select(op->condition, true_value, false_value);
    } else if (const Let *op = expr.as<Let>()) {
        Expr value = forward_accumulation(op->value, tangents, scope);
        std::string fwd_name = op->name + ".fwd";
        scope.push(op->name, Variable::make(op->type, fwd_name));
        Expr body = forward_accumulation(op->body, tangents, scope);
        scope.pop(op->name);
        return Let::make(op->name, op->value,
                Let::make(fwd_name, value, body));
    } else if (const Variable *op = expr.as<Variable>()) {
        if (scope.contains(op->name)) {
            return scope.get(op->name);
        } else {
            return make_zero(op->type);
        }
    } else if (const Call *op = expr.as<Call>()) {
        if (op->is_extern()) {
            if (op->name == "exp_f32") {
                // d/dx exp(f(x)) = exp(f(x)) f'
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return expr * d;
            } else if (op->name == "log_f32" || op->name == "log_f64") {
                // d/dx log(f(x)) = f' / f(x)
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return d / expr;
            } else if (op->name == "sin_f32" || op->name == "sin_f64") {
                // d/dx sin(f(x)) = cos(f(x)) f'
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return cos(op->args[0]) * d;
            } else if (op->name == "asin_f32" || op->name == "asin_f64") {
                // d/dx asin(f(x)) = f' / sqrt(1 - f(x)^2)
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return d / sqrt(1 - pow(op->args[0], 2));
            } else if (op->name == "cos_f32" || op->name == "cos_f64") {
                // d/dx cos(f(x)) = -sin(f(x)) f'
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return -sin(op->args[0]) * d;
            } else if (op->name == "ceil_f32") {
                return make_zero(op->type);
            } else if (op->name == "floor_f32") {
                return make_zero(op->type);
            } else if (op->name == "sqrt_f32") {
                // d/dx f(x)^(0.5) = 0.5 * f(x)^(-0.5) f'
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return (0.5f * d / expr);
            } else if (op->name == "pow_f32") {
                // d/dx pow(f(x), g(x)) = pow(f(x), g(x)-1) *
                //                        (g(x) f'(x) + f(x) log(f(x))g'(x))
                Expr a = forward_accumulation(op->args[0], tangents, scope);
                Expr b = forward_accumulation(op->args[1], tangents, scope);
                return pow(op->args[0], op->args[1] - 1) *
                    (op->args[1] * a +
                     // Special hack: if g' == 0 then even if f == 0 the following term is 0
                     // basically we want -Inf * 0 = 0
                     select(b == 0, make_zero(b.type()), op->args[0] * log(op->args[0]) * b));
            } else if (op->name == "halide_print") {
                return 0.f;
            } else {
                internal_error << "The derivative of " << op->name << " is not implemented.";
            }
        } else if (op->call_type == Call::Image || op->call_type == Call::Halide) {
            auto it = tangents.find(op->name);
            if (it != tangents.end()) {
                Func tangent = it->second;
                return tangent(op->args);
            } else {
                return make_zero(op->type);
            }
        } else {
            internal_assert(op->is_intrinsic());
            if (op->is_intrinsic(Call::abs)) {
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return select(op->args[0] > 0, d, -d);
            } else if (op->is_intrinsic(Call::lerp)) {
                // z = a(x) * (1 - w(x)) + b(x) * w(x)
                // dz/dx = -(w - 1) a' + (b - a) w' + w b'
                Expr a = forward_accumulation(op->args[0], tangents, scope);
                Expr b = forward_accumulation(op->args[1], tangents, scope);
                Expr w = forward_accumulation(op->args[2], tangents, scope);
                return -(op->args[2] - 1) * a + (op->args[1] - op->args[0]) * w + op->args[2] * b;
            } else if (op->is_intrinsic(Call::likely)) {
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return likely(d);
            } else if (op->is_intrinsic(Call::return_second)) {
                Expr d = forward_accumulation(op->args[1], tangents, scope);
                return d;
            } else if (op->is_intrinsic(Call::stringify)) {
                return 0.f;
            } else if (op->is_intrinsic(Call::undef)) {
                return make_zero(op->type);
            } else if (op->is_intrinsic(Call::reinterpret)) {
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                if (is_zero(d)) {
                    return d;
                } else {
                    internal_error << "Can't take a derivative through a reinterpret_cast\n";
                }
            } else {
                internal_error << "The derivative of intrinsic " << op->name << " is not implemented in call: " << Expr(op) << "\n";
            }
        }
    } else {
        return make_zero(expr.type());
    }
    return make_zero(expr.type());
}

Expr forward_accumulation(const Expr &expr,
                          const std::map<std::string, Func> &tangents) {
    Scope<Expr> scope;
    return forward_accumulation(expr, tangents, scope);
}

} // namespace Internal


Derivative propagate_adjoints(const Func &output,
                              const Func &adjoint,
                              const std::vector<std::pair<Expr, Expr>> &output_bounds) {
    user_assert(output.dimensions() == adjoint.dimensions())
      << "output dimensions and adjoint dimensions must match\n";
    user_assert((int)output_bounds.size() == adjoint.dimensions())
      << "output_bounds and adjoint dimensions must match\n";

    Internal::ReverseAccumulationVisitor visitor;
    visitor.propagate_adjoints(output, adjoint, output_bounds);
    return Derivative{visitor.get_adjoint_funcs()};
}

Derivative propagate_adjoints(const Func &output,
                              const Buffer<float> &adjoint) {
    user_assert(output.dimensions() == adjoint.dimensions());
    std::vector<std::pair<Expr, Expr>> bounds;
    for (int dim = 0; dim < adjoint.dimensions(); dim++) {
        bounds.push_back(std::make_pair(Expr(adjoint.min(dim)),
                                        Expr(adjoint.min(dim) + adjoint.extent(dim) - 1)));
    }
    Func adjoint_func("adjoint_func");
    adjoint_func(_) = adjoint(_);
    return propagate_adjoints(output, adjoint_func, bounds);
}

Derivative propagate_adjoints(const Func &output) {
    Func adjoint("adjoint");
    adjoint(output.args()) = Internal::make_one(output.values()[0].type());
    std::vector<std::pair<Expr, Expr>> output_bounds;
    output_bounds.reserve(output.dimensions());
    for (int i = 0; i < output.dimensions(); i++) {
        output_bounds.push_back({0, 0});
    }
    return propagate_adjoints(output, adjoint, output_bounds);
}

Func propagate_tangents(const Func &output,
                        const std::map<std::string, Func> &tangents) {
    // Topologically sort the functions
    std::map<std::string, Internal::Function> env =
        Internal::find_transitive_calls(output.function());
    std::vector<std::string> order =
        Internal::realization_order({output.function()}, env).first;
    std::vector<Func> funcs;
    funcs.reserve(order.size());
    for (const auto &func_name : order) {
        Func func(env[func_name]);
        funcs.push_back(Func(env[func_name]));
    }

    std::vector<Func> transformed_funcs;
    transformed_funcs.reserve(order.size());
    std::map<std::string, Func> updated_tangents = tangents;
    for (const Func &func : funcs) {
        Func transformed_func(func.name() + "_fwd");
        Tuple v = func.values();
        std::vector<Expr> tv;
        for (const Expr &e : v.as_vector()) {
            Expr new_expr = Internal::forward_accumulation(e, updated_tangents);
            //new_expr = print_when(is_nan(new_expr) != 0, new_expr, std::string("NaN founds in ") + transformed_func.name());
            tv.push_back(new_expr);
        }
        transformed_func(func.args()) = Tuple(tv);
        updated_tangents[func.name()] = transformed_func;
        for (int update_id = 0; update_id < func.num_update_definitions(); update_id++) {
            Tuple v = func.update_values(update_id);
            std::vector<Expr> tv;
            for (const Expr &e : v.as_vector()) {
                Expr new_expr = Internal::forward_accumulation(e, updated_tangents);
                //new_expr = print_when(is_nan(new_expr) != 0, new_expr, std::string("NaN founds in ") + transformed_func.name());
                tv.push_back(new_expr);
            }
            transformed_func(func.update_args(update_id)) = Tuple(tv);
            updated_tangents[func.name()] = transformed_func;
        }
        transformed_funcs.push_back(transformed_func);
    }

    return transformed_funcs.back();
}

void print_func(const Func &func, const PrintFuncOptions &options) {
    Internal::debug(0) << "Printing function:" << func.name() << "\n";
    // Topologically sort the functions
    std::map<std::string, Internal::Function> env =
        find_transitive_calls(func.function());
    std::vector<std::string> order = realization_order({func.function()}, env).first;
    std::vector<Func> funcs;
    funcs.reserve(order.size());
    for (const auto &func_name : order) {
        Func func(env[func_name]);
        funcs.push_back(func);
    }

    int lowest_index = 0;
    if (options.depth >= 0) {
        lowest_index = (int)funcs.size() - 1 - options.depth;
    }

    for (int i = (int)funcs.size() - 1; i >= lowest_index; i--) {
        const char *ce = "constant_exterior";
        const char *re = "repeat_edge";
        if (options.ignore_bc && (funcs[i].name().substr(0, strlen(ce)) == std::string(ce) ||
                funcs[i].name().substr(0, strlen(re)) == std::string(re) ||
                funcs[i].name().find("_ce") != std::string::npos)) {
            continue;
        }
        if (options.ignore_non_adjoints && funcs[i].name().find("_d_def__") == std::string::npos) {
            continue;
        }
        Func func = funcs[i];
        Internal::debug(0) << "  funcs[" << i << "]: " << func.name() << "\n";
        for (int update_id = -1; update_id < func.num_update_definitions(); update_id++) {
            Internal::ReductionDomain rdom;
            if (update_id >= 0) {
                Internal::debug(0) << "    update:" << func.name() << "(";
                if (func.update_args(update_id).size() > 0) {
                    Expr e = func.update_args(update_id)[0];
                    for (const auto &it : options.variables) {
                        e = substitute(it.first, it.second, e);
                    }
                    Internal::debug(0) << Internal::simplify(e);
                    for (int i = 1; i < (int)func.update_args(update_id).size(); i++) {
                        Expr e = func.update_args(update_id)[i];
                        for (const auto &it : options.variables) {
                            e = substitute(it.first, it.second, e);
                        }
                        Internal::debug(0) << ", " <<
                            Internal::simplify(e);
                    }
                }
                Internal::debug(0) << ") =";
                auto vals = func.update_values(update_id).as_vector();
                for (auto val : vals) {
                    Expr e = val;
                    for (const auto &it : options.variables) {
                        e = substitute(it.first, it.second, e);
                    }
                    Internal::debug(0) << " " << Internal::simplify(e);
                }
                Internal::debug(0) << "\n";
                //rdom = Internal::extract_rdom(Internal::simplify(func.update_value(update_id)));
            } else {
                Internal::debug(0) << "    " << func.name() << "(";
                if (func.args().size() > 0) {
                    Internal::debug(0) << func.args()[0];
                    for (int i = 1; i < (int)func.args().size(); i++) {
                        Internal::debug(0) << ", " << Internal::simplify(func.args()[i]);
                    }
                }
                Internal::debug(0) << ") =";
                auto vals = func.values().as_vector();
                for (auto val : vals) {
                    Expr e = val;
                    for (const auto &it : options.variables) {
                        e = substitute(it.first, it.second, e);
                    }
                    Internal::debug(0) << " " << Internal::simplify(e);
                }
                Internal::debug(0) << "\n";
                //rdom = Internal::extract_rdom(Internal::simplify(func.value()));
            }

            if (rdom.defined()) {
                Internal::debug(0) << "    RDom:";
                for (int i = 0; i < (int)rdom.domain().size(); i++) {
                    Internal::debug(0) << " (" <<
                        Internal::simplify(rdom.domain()[i].min) << ", " <<
                        Internal::simplify(rdom.domain()[i].extent) << ")";
                }
                Internal::debug(0) << "\n";
            }
        }
    }
}

// Testing code
namespace Internal {

void test_simple_bounds_inference() {
    Var x("x"), y("y");
    int height = 32;
    int width = 16;

    Func input("input");
    input(x, y) = 0.0f;
    Func blur_x("blur_x");
    blur_x(x, y) = input(x, y) + input(x+1, y) + input(x+2, y);
    Func blur_y("blur_y");
    blur_y(x, y) = blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2);

    RDom r(0, width-2, 0, height-2);
    Func f_loss("f_loss");
    f_loss(x) = 0.f;
    f_loss(x) += blur_y(r.x, r.y);

    std::map<std::string, Box> bounds = inference_bounds(f_loss, {{0, 0}});

    internal_assert(equal(bounds[blur_y.name()][0].min, 0))
        << "Expected 0 instead of " << bounds[blur_y.name()][0].min << "\n" ;
    internal_assert(equal(bounds[blur_y.name()][0].max, width-3))
        << "Expected " << width-3  << " instead of " << bounds[blur_y.name()][0].max << "\n" ;
    internal_assert(equal(bounds[blur_y.name()][1].min, 0))
        << "Expected 0 instead of " << bounds[blur_y.name()][1].min << "\n" ;
    internal_assert(equal(bounds[blur_y.name()][1].max, height-3))
        << "Expected " << height-3  << " instead of " << bounds[blur_y.name()][1].max << "\n" ;

    internal_assert(equal(bounds[blur_x.name()][0].min, 0))
        << "Expected 0 instead of " << bounds[blur_x.name()][0].min << "\n" ;
    internal_assert(equal(bounds[blur_x.name()][0].max, width-3))
        << "Expected " << width-3 << " instead of " << bounds[blur_x.name()][0].max << "\n" ;
    internal_assert(equal(bounds[blur_x.name()][1].min, 0))
        << "Expected 0 instead of " << bounds[blur_x.name()][1].min << "\n" ;
    internal_assert(equal(bounds[blur_x.name()][1].max, height-1))
        << "Expected " << height-1 << " instead of " << bounds[blur_x.name()][1].max << "\n" ;

    internal_assert(equal(bounds[input.name()][0].min, 0))
        << "Expected 0 instead of " << bounds[input.name()][0].min << "\n" ;
    internal_assert(equal(bounds[input.name()][0].max, width-1))
        << "Expected " << width-1 << " instead of " << bounds[input.name()][0].max << "\n" ;
    internal_assert(equal(bounds[input.name()][1].min, 0))
        << "Expected 0 instead of " << bounds[input.name()][1].min << "\n" ;
    internal_assert(equal(bounds[input.name()][1].max, height-1))
        << "Expected " << height-1 << " instead of " << bounds[input.name()][1].max << "\n" ;
}

void test_simple_bounds_inference_update() {
    Var x("x");
    Func input("input");
    input(x) = 0.0f;
    Func blur("blur");
    blur(x) = input(x);
    blur(x) += input(x + 1);
    RDom r(0, 2);
    Func f_loss("f_loss");
    f_loss(x) = 0.f;
    f_loss(x) += blur(r.x);

    std::map<std::string, Box> bounds = inference_bounds(f_loss, {{0, 0}});

    internal_assert(equal(bounds[blur.name()][0].min, 0))
        << "Expected 0 instead of " << bounds[blur.name()][0].min << "\n" ;
    internal_assert(equal(bounds[blur.name()][0].max, 1))
        << "Expected 1 instead of " << bounds[blur.name()][0].max << "\n" ;
    internal_assert(equal(bounds[input.name()][0].min, 0))
        << "Expected 0 instead of " << bounds[input.name()][0].min << "\n" ;
    internal_assert(equal(bounds[input.name()][0].max, 2))
        << "Expected 2 instead of " << bounds[input.name()][0].max << "\n" ;
}

#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < 1e-6f) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

void test_scalar() {
    Func x("x");
    x() = 5.f;
    Func y("y");
    y() = x() * x() + 2.f * x() + 5.f;
    Derivative d = propagate_adjoints(y);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> dydx = adjoints[FuncKey{x.name(), -1}].realize();
    // dydx = 2x + 2 = 12
    CMP(dydx(0), 12.f);
}

void test_simple_1d_blur() {
    Var x("x");
    float input_data[] = {1.f, 2.f};
    Buffer<float> input(input_data, 2, "input");
    Func clamped("clamped");
    Expr clamped_x = Halide::clamp(x, 0, input.width() - 1);
    clamped(x) = input(clamped_x);
    Func blur("blur");
    blur(x) = clamped(x) + clamped(x + 1);
    RDom r(0, 2);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += blur(r.x) * blur(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> blur_buf = blur.realize(2);
    // d loss / d blur = 2 * blur(x)
    Buffer<float> d_blur_buf = adjoints[FuncKey{blur.name(), -1}].realize(2);

    CMP(d_blur_buf(0), 2 * blur_buf(0));
    CMP(d_blur_buf(1), 2 * blur_buf(1));
    Buffer<float> d_clamped_buf = adjoints[FuncKey{clamped.name(), -1}].realize(2);
    CMP(d_clamped_buf(0), d_blur_buf(0));
    CMP(d_clamped_buf(1), d_blur_buf(0) + d_blur_buf(1));
}

void test_simple_1d_blur_no_clamp() {
    Var x("x");
    float input_data[] = {1.f, 2.f};
    Buffer<float> input(input_data, 2, "input");
    Func blur("blur");
    blur(x) = input(x) + input(x + 1);
    RDom r(0, 1);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += blur(r.x) * blur(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> blur_buf = blur.realize(1);
    // d loss / d blur = 2 * blur(x)
    Buffer<float> d_blur_buf = adjoints[FuncKey{blur.name(), -1}].realize(1);

    CMP(d_blur_buf(0), 2 * blur_buf(0));
    Buffer<float> d_clamped_buf = adjoints[FuncKey{input.name(), -1}].realize(1);
    CMP(d_clamped_buf(0), d_blur_buf(0));
}

void test_simple_2d_blur() {
    Var x("x"), y("y");
    float input_data[] = {
        0.f, 1.f, 0.f, 0.f, 0.f,
        1.f, 1.f, 1.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f
    };
    Buffer<float> input(input_data, 5, 5, "input");
    Func clamped("clamped");
    Expr clamped_x = Halide::clamp(x, 0, input.width()-1);
    Expr clamped_y = Halide::clamp(y, 0, input.height()-1);
    clamped(x, y) = input(clamped_x, clamped_y);
    Func blur_x("blur_x");
    blur_x(x, y) = clamped(x, y) + clamped(x + 1, y) + clamped(x + 2, y);
    Func blur_y("blur_y");
    blur_y(x, y) = blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2);

    RDom r(0, 5, 0, 5);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += blur_y(r.x, r.y) * blur_y(r.x, r.y);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> blur_y_buf = blur_y.realize(5, 5);
    // d loss / d blur_y = 2 * blur_y(x, y)
    Buffer<float> d_blur_y_buf = adjoints[FuncKey{blur_y.name(), -1}].realize(5, 5);
    const float eps = 1e-6;
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            float target = 2 * blur_y_buf(x, y);
            float diff = fabs(d_blur_y_buf(x, y) - target);
            internal_assert(diff < eps)
                << "Expected d_blur_y(" << x << ", " << y << ") to be " <<
                    target << " instead of " << d_blur_y_buf(x, y) << "\n" ;
        }
    }
    // d loss / d blur_x = d blur_y(x, y) + d blur_y(x, y - 1) + d blur_y(x, y - 2)
    Buffer<float> d_blur_x_buf = adjoints[FuncKey{blur_x.name(), -1}].realize(5, 5);
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            float target = d_blur_y_buf(x, y);
            if (y >= 1) {
                target += d_blur_y_buf(x, y - 1);
            }
            if (y >= 2) {
                target += d_blur_y_buf(x, y - 2);
            }
            float diff = fabs(d_blur_x_buf(x, y) - target);
            internal_assert(diff < eps)
                << "Expected d_blur_x(" << x << ", " << y << ") to be " <<
                target << " instead of " << d_blur_x_buf(x, y) << "\n" ;
        }
    }
    Buffer<float> d_clamped = adjoints[FuncKey{clamped.name(), -1}].realize(5, 5);
    // d loss / d clamped = d blur_x(x, y) + d blur_x(x - 1, y) + d blur_x(x - 2, y)
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            float target = d_blur_x_buf(x, y);
            if (x >= 1) {
                target += d_blur_x_buf(x - 1, y);
            }
            if (x >= 2) {
                target += d_blur_x_buf(x - 2, y);
            }
            float diff = fabs(d_clamped(x, y) - target);
            internal_assert(diff < eps)
                << "Expected d_clamped(" << x << ", " << y << ") to be " <<
                target << " instead of " << d_clamped(x, y) << "\n" ;
        }
    }
}

void test_update() {
    Var x("x");
    float input_data[] = {1.f, 2.f};
    Buffer<float> input(input_data, 2, "input");
    Func clamped("clamped");
    Expr clamped_x = Halide::clamp(x, 0, input.width() - 1);
    clamped(x) = input(clamped_x);
    Func blur("blur");
    blur(x) = clamped(x);
    blur(x) += clamped(x + 1);
    RDom r(0, 2);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += blur(r.x) * blur(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> blur_buf = blur.realize(2);
    // d loss / d blur = 2 * blur(x)
    Buffer<float> d_blur_buf = adjoints[FuncKey{blur.name(), -1}].realize(2);

    CMP(d_blur_buf(0), 2 * blur_buf(0));
    CMP(d_blur_buf(1), 2 * blur_buf(1));
    Buffer<float> d_clamped_buf = adjoints[FuncKey{clamped.name(), -1}].realize(2);
    CMP(d_clamped_buf(0), d_blur_buf(0));
    CMP(d_clamped_buf(1), d_blur_buf(0) + d_blur_buf(1));
}

void test_rdom_conv() {
    Var x("x");
    float input_data[] = {1.f, 2.f, 3.f, 4.f};
    Buffer<float> input(input_data, 4, "input");
    Func clamped("clamped");
    Expr clamped_x = Halide::clamp(x, 0, input.width() - 1);
    clamped(x) = input(clamped_x);
    float kernel_data[] = {1.f, 1.f};
    Buffer<float> kernel(kernel_data, 2, "kernel");
    Func kernel_func("kernel_func");
    kernel_func(x) = kernel(x);
    Func convolved("convolved");
    RDom support(0, 2);
    convolved(x) = 0.f;
    convolved(x) += clamped(x + support.x) * kernel_func(support.x);
    RDom r(0, 4);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += convolved(r.x) * convolved(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;
    Buffer<float> convolved_buf = convolved.realize(4);
    // d loss / d blur = 2 * blur(x)
    Buffer<float> d_convolved_buf = adjoints[FuncKey{convolved.name(), -1}].realize(4);

    for (int i = 0; i < 4; i++) {
        CMP(d_convolved_buf(i), 2 * convolved_buf(i));
    }
    // d loss / d clamped = d_convolved convolve with flipped kernel
    Buffer<float> d_clamped_buf = adjoints[FuncKey{clamped.name(), -1}].realize(4);
    for (int i = 0; i < 4; i++) {
        float target = d_convolved_buf(i) * kernel_data[0];
        if (i >= 1) {
            target += d_convolved_buf(i - 1) * kernel_data[1];
        }
        CMP(d_clamped_buf(i), target);
    }
    // loss = (k0 + 2k1)^2 + (2k0 + 3k1)^2 + (3k0 + 4k1)^2 + (4k0 + 4k1)^2
    //      = k0^2 + 4k0k1 + 4k1^2 + 4k0^2 + 12 k0k1 + 9k1^2 + 9k0^2 + 24 k0k1 + 16 k1^2 + 16k0^2 + 32k0k1 + 16k1^2
    //      = 30 k0^2 + 72 k0k1 + 45 k1^2
    // d loss / d kernel(0) = 2 * 30 + 72 = 132
    // d loss / d kernel(1) = 2 * 45 + 72 = 162
    Buffer<float> d_kernel_buf = adjoints[FuncKey{kernel_func.name(), -1}].realize(2);
    CMP(d_kernel_buf(0), 132);
    CMP(d_kernel_buf(1), 162);
}

void test_1d_to_2d() {
    Var x("x"), y("y");
    float input_data[] = {1.f, 2.f};
    Buffer<float> input(input_data, 2, "input");
    Func f_output("output");
    f_output(x, y) = input(y);

    RDom r(0, 2, 0, 2);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += f_output(r.x, r.y) * f_output(r.x, r.y);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    // loss = 2i0^2 + 2i1^2
    // d loss / d i0 = 4i0 = 4
    // d loss / d i1 = 4i1 = 8

    Buffer<float> d_output = adjoints[FuncKey{f_output.name(), -1}].realize(2, 2);
    CMP(d_output(0, 0), 2);
    CMP(d_output(1, 0), 2);
    CMP(d_output(0, 1), 4);
    CMP(d_output(1, 1), 4);

    Buffer<float> d_input = adjoints[FuncKey{input.name(), -1}].realize(2);
    CMP(d_input(0), 4);
    CMP(d_input(1), 8);
}

void test_linear_interpolation() {
    Var x("x");
    float input_data0[] = {0.3f, 1.8f};
    float input_data1[] = {1.0f, 2.0f, 4.0f};
    Buffer<float> input0(input_data0, 2, "input0");
    Buffer<float> input1(input_data1, 3, "input1");
    Func f_input0("f_input0");
    Expr clamped_x0 = Halide::clamp(x, 0, input0.width() - 1);
    f_input0(x) = input0(clamped_x0);
    Func f_input1("f_input1");
    Expr clamped_x1 = Halide::clamp(x, 0, input1.width() - 1);
    f_input1(x) = input1(clamped_x1);
    Expr gx = f_input0(x);
    Expr fx = cast<int>(clamp(floor(f_input0(x)), 0.f, 1.f));
    Expr cx = fx + 1;
    Expr wx = gx - fx;
    Func f_interpolate("interpolate");
    Expr f1 = f_input1(fx);
    f_interpolate(x) = f_input1(fx) * (1.f - wx) + f_input1(cx) * wx;

    RDom r(0, 2);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += f_interpolate(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    // f_interpolate = {i1[0] * (1 - (i0[0] - floor(i0[0]))) +
    //                  i1[1] * (i0[0] - floor(i0[0])),
    //                  i1[1] * (1 - (i0[1] - floor(i0[1]))) +
    //                  i1[2] * (i0[1] - floor(i0[1]))}
    // loss = f_interpolate[0] + f_interpolate[1]
    // d loss / d i0[0] = -i1[0] + i1[1] = 1
    // d loss / d i0[1] = -i1[1] + i1[2] = 2
    // d loss / d i1[0] = 1 - (i0[0] - floor(i0[0]))
    // d loss / d i1[1] = (i0[0] - floor(i0[0])) +
    //                    (1 - (i0[1] - floor(i0[1])))
    // d loss / d i1[2] = i0[1] - floor(i0[1])

    Buffer<float> interpolate = f_interpolate.realize(2);
    CMP(interpolate(0), 1.3f);
    CMP(interpolate(1), 3.6f);

    Buffer<float> d_input_0 = adjoints[FuncKey{f_input0.name(), -1}].realize(2);
    CMP(d_input_0(0), 1.f);
    CMP(d_input_0(1), 2.f);

    Buffer<float> d_input_1 = adjoints[FuncKey{f_input1.name(), -1}].realize(3);
    CMP(d_input_1(0), 0.7f);
    CMP(d_input_1(1), 0.5f);
    CMP(d_input_1(2), 0.8f);
}

void test_linear_interpolation_2d() {
    Var x("x"), y("y");
    float input_data0[] = {0.3f, 1.8f};
    float input_data1[] = {1.0f, 2.0f, 4.0f};
    Buffer<float> input0(input_data0, 2, 1, "input0");
    Buffer<float> input1(input_data1, 3, 1, "input1");
    Func f_input0("f_input0");
    Expr clamped_x0 = Halide::clamp(x, 0, input0.width() - 1);
    Expr clamped_y0 = Halide::clamp(y, 0, input0.height() - 1);
    f_input0(x, y) = input0(clamped_x0, clamped_y0);
    Func f_input1("f_input1");
    Expr clamped_x1 = Halide::clamp(x, 0, input1.width() - 1);
    Expr clamped_y1 = Halide::clamp(y, 0, input1.height() - 1);
    f_input1(x, y) = input1(clamped_x1, clamped_y1);
    Expr gx = f_input0(x, y);
    Expr fx = cast<int>(clamp(floor(f_input0(x, y)), 0.f, 1.f));
    Expr cx = fx + 1;
    Expr wx = gx - fx;
    Func f_interpolate("interpolate");
    Expr f1 = f_input1(fx, y);
    f_interpolate(x, y) = f_input1(fx, y) * (1.f - wx) + f_input1(cx, y) * wx;

    RDom r(0, 2, 0, 1);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += f_interpolate(r.x, r.y);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    // Same as test_linear_interpolation()
    Buffer<float> interpolate = f_interpolate.realize(2, 1);
    CMP(interpolate(0, 0), 1.3f);
    CMP(interpolate(1, 0), 3.6f);

    Buffer<float> d_input_0 = adjoints[FuncKey{f_input0.name(), -1}].realize(2, 1);
    CMP(d_input_0(0, 0), 1.f);
    CMP(d_input_0(1, 0), 2.f);

    Buffer<float> d_input_1 = adjoints[FuncKey{f_input1.name(), -1}].realize(3, 1);
    CMP(d_input_1(0, 0), 0.7f);
    CMP(d_input_1(1, 0), 0.5f);
    CMP(d_input_1(2, 0), 0.8f);
}

void test_sparse_update() {
    Var x("x");
    float input_data[] = {1.0f, 2.0f, 3.0f};
    Buffer<float> input(input_data, 3, "input");
    Func f_input("f_input");
    f_input(x) = input(x);
    Func f_output("f_output");
    f_output(x) = f_input(x);
    f_output(1) = 0.f;
    // XXX: if we do input(1) Halide returns a float
    // which means it is impossible to propagate to input, so we need a surrogate
    f_output(2) = f_input(1);

    Func f_loss("f_loss");
    RDom r(0, 3);
    f_loss() = 0.f;
    f_loss() += f_output(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> d_input = adjoints[FuncKey{input.name(), -1}].realize(3);
    CMP(d_input(0), 1.0f);
    CMP(d_input(1), 1.0f);
    CMP(d_input(2), 0.0f);
}

void test_rdom_update() {
    Var x("x");
    float input_data[] = {1.0f, 2.0f, 3.0f};
    Buffer<float> input(input_data, 3, "input");
    Func f_output("f_output");
    RDom r0(1, 2), r1(3, 4);
    f_output(x) = input(x);
    f_output(r0) = input(r0 - 1);
    f_output(r1) = 0.f;

    Func f_loss("f_loss");
    RDom r_target(0, 5);
    f_loss() = 0.f;
    f_loss() += f_output(r_target);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> d_input = adjoints[FuncKey{input.name(), -1}].realize(3);
    CMP(d_input(0), 2.0f);
    CMP(d_input(1), 1.0f);
    CMP(d_input(2), 0.0f);
}

void test_repeat_edge() {
    Var x("x");
    float input_data[] = {1.f, 2.f};
    Buffer<float> input(input_data, 2, "input");
    Func clamped = BoundaryConditions::repeat_edge(input);
    Func blur("blur");
    blur(x) = clamped(x) + clamped(x + 1);
    RDom r(0, 3);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += blur(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;
    // loss = (i0 + i1) + (i1 + i1) + (i1 + i1) = i0 + 5 * i1

    Buffer<float> d_blur_buf = blur.realize(3);
    Buffer<float> d_input_buf = adjoints[FuncKey{input.name(), -1}].realize(2);
    // d loss / d i0 = 1
    // d loss / d i1 = 5
    CMP(d_input_buf(0), 1.f);
    CMP(d_input_buf(1), 5.f);
}

void test_second_order() {
    Var x("x");
    float input_data[] = {1.f};
    Buffer<float> input(input_data, 1, "input");
    Func polynomial("polynomial");
    // x^2 + 3x + 4.f
    polynomial(x) = input(x) * input(x) + 3.f * input(x) + 4.f;
    Derivative d = propagate_adjoints(polynomial);
    Func d_input = d.adjoints[FuncKey{input.name(), -1}];
    Derivative d2 = propagate_adjoints(d_input);
    Func d2_input = d2.adjoints[FuncKey{input.name(), -1}];

    Buffer<float> buf = d_input.realize(1);
    Buffer<float> buf2 = d2_input.realize(1);
    // d/dx = 2x + 3
    CMP(buf(0), 5.f);

    // d^2/dx^2 = 2
    CMP(buf2(0), 2.f);
}

void test_second_order_conv() {
    Var x("x");
    Buffer<float> input(10, "input");
    for (int i = 0; i < 10; i++) {
        input(i) = float(i) / 10.f;
    }
    Buffer<float> target(10, "target");
    for (int i = 0; i < 10; i++) {
        target(i) = float(i + 1) / 10.f;
    }
    Buffer<float> kernel(3, "kernel");
    kernel(0) = kernel(1) = kernel(2) = 1.f;
    Func input_re = BoundaryConditions::repeat_edge(input);
    RDom rc(0, 3);
    Func conv("conv");
    conv(x) = 0.f;
    conv(x) += input_re(x + rc - 1) * kernel(rc);
    RDom rl(0, 9);
    Func loss0("loss0");
    loss0() = 0.f;
    loss0() += pow(conv(rl) - target(rl), 2.f);
    Derivative d = propagate_adjoints(loss0);
    Func d_input = d(input);
    Func loss1("loss1");
    loss1() = 0.f;
    loss1() += d_input(rl);
    Derivative d2 = propagate_adjoints(loss1);

    Buffer<float> conv_buf = conv.realize(9);
    Buffer<float> d_conv_buf = d(conv).realize(9);
    // d_conv(x) = 2 * (conv(x) - target(x))
    for (int i = 0; i < 9; i++) {
        CMP(d_conv_buf(i), 2.f * (conv_buf(i) - target(i)));
    }
    // d_input(x) = d_conv(x + 1) + d_conv(x) + d_conv(x - 1)
    Buffer<float> d_input_buf = d_input.realize(10);
    CMP(d_input_buf(0), d_conv_buf(0) + d_conv_buf(1));
    for (int i = 1; i < 8; i++) {
        CMP(d_input_buf(i), d_conv_buf(i+1) + d_conv_buf(i) + d_conv_buf(i-1));
    }
    CMP(d_input_buf(8), d_conv_buf(7) + d_conv_buf(8));
    CMP(d_input_buf(9), d_conv_buf(8));
    Buffer<float> d2_conv_buf = d2(conv).realize(9);
    // d2_conv(x) = 6
    for (int i = 0; i < 8; i++) {
        CMP(d2_conv_buf(i), 6.f);
    }
    CMP(d2_conv_buf(8), 4.f);
    // d2_input(x) = d2_conv(x + 1) + d2_conv(x) + d2_conv(x - 1)
    Buffer<float> d2_input_buf = d2(input).realize(10);
    CMP(d2_input_buf(0), d2_conv_buf(0) + d2_conv_buf(1));
    for (int i = 1; i <= 7; i++) {
        CMP(d2_input_buf(i), d2_conv_buf(i) + d2_conv_buf(i + 1) + d2_conv_buf(i - 1));
    }
    CMP(d2_input_buf(8), d2_conv_buf(8) + d2_conv_buf(7));
    CMP(d2_input_buf(9), d2_conv_buf(7));
}

void test_implicit_vars() {
    Var x("x");
    float input_data[] = {1.f, 2.f};
    Buffer<float> input(input_data, 2, "input");
    Func copy("copy");
    copy(_) = input(_);
    RDom r(0, 2);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += copy(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> d_input_buf = adjoints[FuncKey{input.name(), -1}].realize(2);
    CMP(d_input_buf(0), 1.f);
    CMP(d_input_buf(1), 1.f);
    Buffer<float> d_copy_buf = adjoints[FuncKey{copy.name(), -1}].realize(2);
    CMP(d_copy_buf(0), 1.f);
    CMP(d_copy_buf(1), 1.f);
}

void test_tuple() {
    Var x("x");
    float input_data[] = {1.f, 2.f, 3.f};
    Buffer<float> input(input_data, 3, "input");
    Func tuple("tuple");
    tuple(x) = Tuple(input(x), input(x + 1));
    tuple(x) += Tuple(1.f, 1.f);
    Func reduce("reduce");
    reduce(x) = tuple(x)[0] + tuple(x)[1];
    RDom r(0, 2);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += reduce(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;
    // tuple(0) = {1, 2}
    // tuple(1) = {2, 3}
    // reduce(0) = 3
    // reduce(1) = 5
    // loss = reduce(0) + reduce(1)
    //      = tuple(0)[0] + tuple(0)[1] + tuple(1)[0] + tuple(1)[1]
    //      = input(0) + input(1) * 2 + input(2)

    Realization d_tuple_buf = adjoints[FuncKey{tuple.name(), -1}].realize(2);
    Buffer<float> d_tuple_buf_0 = d_tuple_buf[0];
    Buffer<float> d_tuple_buf_1 = d_tuple_buf[1];
    CMP(d_tuple_buf_0(0), 1.f);
    CMP(d_tuple_buf_0(1), 1.f);
    CMP(d_tuple_buf_1(0), 1.f);
    CMP(d_tuple_buf_1(1), 1.f);

    Buffer<float> d_input_buf = adjoints[FuncKey{input.name(), -1}].realize(3);
    CMP(d_input_buf(0), 1.f);
    CMP(d_input_buf(1), 2.f);
    CMP(d_input_buf(2), 1.f);
}

void test_floor_ceil() {
    Var x("x");
    float input_data[] = {1.f, 2.f, 3.f};
    Buffer<float> input(input_data, 3, "input");
    Func floor_output("floor_output");
    floor_output(x) = input(cast<int>(floor(x / 4.f)));
    Func ceil_output("ceil_output");
    ceil_output(x) = input(cast<int>(ceil(x / 4.f)));
    Func output("output");
    output(x) = ceil_output(x) + floor_output(x);
    RDom r(0, 8);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += output(r.x);
    Derivative d = propagate_adjoints(f_loss);
    // floor_output(0~3) == input[0]
    // floor_output(4~7) == input[1]
    // ceil_output(0) == input[0]
    // ceil_output(1~4) == input[1]
    // ceil_output(5~7) = input[2]
    Buffer<float> d_input_buf = d(input).realize(3);

    CMP(d_input_buf(0), 5.f);
    CMP(d_input_buf(1), 8.f);
    CMP(d_input_buf(2), 3.f);
}

void test_downsampling() {
    Var x("x");
    Buffer<float> input(10);
    for (int i = 0; i < 10; i++) {
        input(i) = float(i);
    }
    Func output("output");
    RDom r(0, 4);
    output(x) = 0.f;
    output(x) += input(4 * x + r);
    RDom r_loss(0, 2);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += output(r_loss);
    Derivative d = propagate_adjoints(f_loss);
    // output(0) = \sum input(0~4)
    // output(1) = \sum input(5~8)
    Buffer<float> d_input_buf = d(input).realize(10);

    for (int i = 0; i < 8; i++) {
        CMP(d_input_buf(i), 1.f);
    }
    CMP(d_input_buf(8), 0.f);
    CMP(d_input_buf(9), 0.f);
}

void test_transpose() {
    Var x("x"), y("y");
    Buffer<float> input(5, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            input(i, j) = float(i + j);
        }
    }
    Buffer<float> target(5, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            target(i, j) = float(i * j);
        }
    }
    Func output("output");
    output(x, y) = input(y, x);
    RDom r(0, 5, 0, 5);
    Func loss("loss");
    loss() = 0.f;
    loss() += pow(output(r.x, r.y) - target(r.x, r.y), 2);
    Derivative d = propagate_adjoints(loss);
    Func d_input = d(input);
    Buffer<float> d_input_buf = d_input.realize(5, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            CMP(d_input_buf(i, j), 2.f * (input(i, j) - target(j, i)));
        }
    }
}

void test_forward() {
    Var x("x");
    Buffer<float> input(10);
    for (int i = 0; i < 10; i++) {
        input(i) = float(i);
    }
    Func output("output");
    RDom r(0, 2);
    output(x) = 0.f;
    output(x) += input(x + r);
    Func d_input("d_input");
    d_input(x) = 1.f;
    Func d_output = propagate_tangents(output, {{input.name(), d_input}});
    // d_output(x) = \sum d_input(x + r)
    Buffer<float> d_output_buf = d_output.realize(5);

    for (int i = 0; i < 5; i++) {
        CMP(d_output_buf(i), 2.f);
    }
}

void test_reverse_forward() {
    Var x("x");
    Buffer<float> input(10, "input");
    for (int i = 0; i < 10; i++) {
        input(i) = float(i);
    }
    Buffer<float> target(10, "target");
    for (int i = 0; i < 10; i++) {
        target(i) = float(i + 1);
    }
    Buffer<float> kernel(2, "kernel");
    kernel(0) = kernel(1) = 1.f;
    Func input_re = BoundaryConditions::repeat_edge(input);
    Func output("output");
    RDom r(0, 2);
    output(x) = 0.f;
    output(x) += input_re(x + r) * kernel(r);
    RDom ro(0, 9);
    Func loss("loss");
    loss() = 0.f;
    Expr diff = output(ro) - target(ro);
    loss() += diff * diff;
    Derivative d = propagate_adjoints(loss);
    Buffer<float> output_buf = output.realize(9);
    Func d_output = d(output);
    // d_output(x) = 2 * (output(x) - target(x))
    Buffer<float> d_output_buf = d_output.realize(9);
    for (int i = 0; i < 9; i++) {
        CMP(d_output_buf(i), 2.f * (output_buf(i) - target(i)));
    }
    Func d_input = d(input);
    Buffer<float> d_input_buf = d_input.realize(10);
    // d_input(x) = d_output(x) + d_output(x - 1)
    CMP(d_input_buf(0), d_output_buf(0));
    for (int i = 1; i < 9; i++) {
        CMP(d_input_buf(i), d_output_buf(i) + d_output_buf(i - 1));
    }
    CMP(d_input_buf(9), d_output_buf(8));
    Func d2_output = propagate_tangents(d_output, {{input.name(), d_input}});
    Buffer<float> d2_output_buf = d2_output.realize(9);
    // d2_output(x) = 2 * (d_input(x) + d_input(x + 1))
    for (int i = 0; i < 9; i++) {
        CMP(d2_output_buf(i), 2.f * (d_input_buf(i) + d_input_buf(i + 1)));
    }
    Func d2_input = propagate_tangents(d_input, {{input.name(), d_input}});
    Buffer<float> d2_input_buf = d2_input.realize(10);
    // d2_input(x) = d2_output(x) + d2_output(x - 1)
    CMP(d2_input_buf(0), d2_output_buf(0));
    for (int i = 1; i < 9; i++) {
        CMP(d2_input_buf(i), d2_output_buf(i) + d2_output_buf(i - 1));
    }
    CMP(d2_input_buf(9), d2_output_buf(8));
}

void derivative_test() {
    test_simple_bounds_inference();
    test_simple_bounds_inference_update();
    test_scalar();
    test_simple_1d_blur();
    test_simple_1d_blur_no_clamp();
    test_simple_2d_blur();
    test_update();
    test_rdom_conv();
    test_1d_to_2d();
    test_linear_interpolation();
    test_linear_interpolation_2d();
    test_sparse_update();
    test_rdom_update();
    test_repeat_edge();
    test_second_order();
    //test_second_order_conv();
    test_implicit_vars();
    test_tuple();
    test_floor_ceil();
    test_downsampling();
    test_transpose();
    test_forward();
    test_reverse_forward();
    debug(0) << "Derivative test passed\n";
}

} // namespace Internal
} // namespace Halide
