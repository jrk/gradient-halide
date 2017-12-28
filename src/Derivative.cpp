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
            if (is_final_output) {
                adjoint_func(func.args()) = adjoint(func.args());
            } else {
                if (func.values().size() == 1) {
                    adjoint_func(func.args()) = 0.f;
                } else {
                    std::vector<Expr> init(func.values().size(), Expr(0.f));
                    adjoint_func(func.args()) = Tuple(init);
                }
            }
            FuncKey func_key{func.name(), update_id};
            adjoint_funcs[func_key] = adjoint_func;
        }
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

            // Set up boundary condition if this is the first visit
            if (update_id == func.num_update_definitions() - 1) {
                Func &adjoint_func = adjoint_funcs[func_key];
                const Box &bounds = func_bounds[func.name()];
                if (adjoint_func.values().size() == 1) {
                    adjoint_func = BoundaryConditions::constant_exterior(
                            adjoint_func, 0.f, box_to_vector(bounds),
			    adjoint_func.name() + "_ce");
                } else {
                    std::vector<Expr> values(adjoint_func.values().size(), Expr(0.f));
                    adjoint_func = BoundaryConditions::constant_exterior(
                            adjoint_func, Tuple(values), box_to_vector(bounds),
			    adjoint_func.name() + "_ce");
                }
                adjoint_funcs[func_key] = adjoint_func;
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
                // Check if next_args are the same as update_args
                // e.g.
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
                    next_adjoint_func(update_args) = 0.f;
                } else {
                    std::vector<Expr> init(func.values().size(), Expr(0.f));
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
                for (const auto &var : func.args()) {
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
    accumulate(op->a, select(op->a <= op->b, adjoint, 0.f));
    // d/db min(a, b) = b <= a ? 1 : 0
    accumulate(op->b, select(op->b <= op->a, adjoint, 0.f));
}

void ReverseAccumulationVisitor::visit(const Max *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/da max(a, b) = a >= b ? 1 : 0
    accumulate(op->a, select(op->a >= op->b, adjoint, 0.f));
    // d/db max(a, b) = b >= a ? 1 : 0
    accumulate(op->b, select(op->b >= op->a, adjoint, 0.f));
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
    accumulate(op->true_value, select(op->condition, adjoint, 0.f));
    // d/dc select(a, b, c) = select(a, 0, 1)
    accumulate(op->false_value, select(op->condition, 0.f, adjoint));
}

void ReverseAccumulationVisitor::visit(const Call *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];
    // Math functions
    if (op->is_extern()) {
      if (op->name == "exp_f32") {
          // d/dx exp(x) = exp(x)
          for (size_t i = 0; i < op->args.size(); i++) {
              accumulate(op->args[i], adjoint * exp(op->args[i]));
          }
      } else if (op->name == "sin_f32") {
          // d/dx sin(x) = cos(x)
          for (size_t i = 0; i < op->args.size(); i++) {
              accumulate(op->args[i], adjoint * cos(op->args[i]));
          }
      } else if (op->name == "cos_f32") {
          // d/dx cos(x) = -sin(x)
          for (size_t i = 0; i < op->args.size(); i++) {
              accumulate(op->args[i], - adjoint * sin(op->args[i]));
          }
      } else if (op->name == "ceil_f32") {
          // TODO: d/dx = dirac(n) for n in Z ...
          for (size_t i = 0; i < op->args.size(); i++) {
              accumulate(op->args[i], 0.0f);
          }
      } else if (op->name == "floor_f32") {
          // TODO: d/dx = dirac(n) for n in Z ...
          for (size_t i = 0; i < op->args.size(); i++) {
              accumulate(op->args[i], 0.0f);
          }
      } else if (op->name == "sqrt_f32") {
          accumulate(op->args[0], adjoint * 0.5f / sqrt(op->args[0]));
      } else if (op->name == "pow_f32") {
          accumulate(op->args[0], adjoint * op->args[1] * pow(op->args[0], op->args[1] - 1.f));
          accumulate(op->args[1], adjoint * pow(op->args[0], op->args[1]) * log(op->args[0]));
      } else {
          internal_error << "The derivative of " << op->name << " is not implemented.";
      }
    } else if (op->call_type == Call::Halide) { // Halide function call
        Function func(op->func);
        // We are scattering to this function
        // debug(0) << "Scattering to " << func.name() << "\n";

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
        FuncKey func_key = func.name() != current_func.name() ?
                           FuncKey{func.name(), func.updates().size() - 1} :
                           FuncKey{func.name(), current_update_id - 1};
        assert(adjoint_funcs.find(func_key) != adjoint_funcs.end());
        Func& func_to_update = adjoint_funcs[func_key];

        bool debug_flag = false;//func_to_update.name() == "f_input_0_d_def__";

        if (debug_flag) {
            debug(0) << "lhs is:";
            for (const auto &arg : lhs) {
                debug(0) << " " << arg;
            }
            debug(0) << "\n";
            debug(0) << "adjoint is:" << simplify(adjoint) << "\n";
        }

        // Gather argument & bounds information
        // current_args are the pure variables
        // current_update_args are the actual updates at left hand side
        std::vector<Var> current_args = current_func.args();
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

        std::vector<bool> canonicalized(lhs.size(), false);
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
        std::set<std::string> canonicalized_vars;
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
            canonicalized_vars.insert(func_to_update.args()[i].name());
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
                std::vector<std::string> variables = gather_variables(lhs_arg, current_func.function().args());
                RDom r(bounds);
                for (int var_id = 0; var_id < (int)variables.size(); var_id++) {
                    for (int arg_id = 0; arg_id < (int)current_args.size(); arg_id++) {
                        if (current_args[arg_id].name() == variables[var_id] &&
                                canonicalized_vars.find(current_args[arg_id].name()) == canonicalized_vars.end()) {
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

        std::vector<Var> func_to_update_args = func_to_update.args();
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
                const Box &target_bounds = func_bounds[func.name()];
                Interval t_interval = target_bounds[i];
                t_interval.min = simplify(t_interval.min);
                t_interval.max = simplify(t_interval.max);
                Interval r_interval(simplify(rvar.min),
                                    simplify(rvar.min + rvar.extent - 1));
                if (equal(r_interval.min, t_interval.min) &&
                        equal(r_interval.max, t_interval.max)) {
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

        // Final simplification pass
        if (debug_flag) {
            Scope<Interval> bounds;
            if (merged_r.defined()) {
                for (int d = 0; d < merged_r.dimensions(); d++) {
                    bounds.push(merged_r[d].name(),
                        Interval(merged_r[d].min(), merged_r[d].min() + merged_r[d].extent() - 1));
                }
            }
            adjoint = simplify(adjoint, true, bounds);
        }

        // TODO: maybe do some analysis on lhs to avoid applying boundary conditions to
        //       function calls in adjoint
        if (func_to_update.values().size() == 1) {
            func_to_update(lhs) += adjoint;
        } else {
            func_to_update(lhs)[op->value_index] += adjoint;
        }

        if (debug_flag) {
            debug(0) << "func_to_update.name():" << func_to_update.name() << "\n";
            debug(0) << "lhs after canonicalization:";
            for (const auto &arg : lhs) {
                debug(0) << " " << arg;
            }
            debug(0) << "\n";
            debug(0) << "adjoint after canonicalization:" << simplify(adjoint) << "\n";
        }
        // print_func(Func(func), true, false, false);
        // print_func(func_to_update, true, true, false);
    } else if (op->call_type != Call::Image) {  // Image loads should not be propagated
        // op->call_type is Call::Intrinsic or Call::PureIntrinsic
        if (op->is_intrinsic(Call::abs)) {
            accumulate(op->args[0], adjoint*select(op->args[0] > 0, 1.0f, -1.0f));
        } else if (op->is_intrinsic(Call::lerp)) {
            // z = x * (1 - w) + y * w
            // dz/dx = 1 - w
            // dz/dy = w
            // dz/dw = y - x
            accumulate(op->args[0], adjoint * (1.f - op->args[2]));
            accumulate(op->args[1], adjoint * op->args[2]);
            accumulate(op->args[2], adjoint * (op->args[1] - op->args[0]));
        } else if (op->is_intrinsic(Call::likely)) {
            accumulate(op->args[0], adjoint);
        } else if (op->is_intrinsic(Call::undef)) {
            // do nothing
        } else {
            internal_error << "The derivative of intrinsic " << op->name << " is not implemented.";
        }
    }
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
    adjoint(output.args()) = 1.f;
    std::vector<std::pair<Expr, Expr>> output_bounds;
    output_bounds.reserve(output.dimensions());
    for (int i = 0; i < output.dimensions(); i++) {
        output_bounds.push_back({0, 0});
    }
    return propagate_adjoints(output, adjoint, output_bounds);
}

void simple_autoschedule(std::vector<Func> &outputs,
                         const std::map<std::string, int> &parameters,
                         const std::vector<std::vector<std::pair<int, int>>> &output_bounds,
                         const SimpleAutoscheduleOptions &options,
                         const std::set<std::string> &skip_functions) {
    using namespace Internal;
    std::vector<FuncBounds> output_bounds_expr;
    for (const auto &bounds : output_bounds) {
        FuncBounds func_bounds;
        for (const auto &bound : bounds) {
            func_bounds.push_back(std::make_pair<Expr, Expr>(bound.first, bound.second));
        }
        output_bounds_expr.push_back(func_bounds);
    }
    std::map<std::string, Box> func_bounds = inference_bounds(outputs, output_bounds_expr);
    std::vector<Function> output_functions;
    output_functions.reserve(outputs.size());
    for (const auto &func : outputs) {
        output_functions.push_back(func.function());
    }
    std::map<std::string, Function> env;
    for (const auto &func : output_functions) {
        std::map<std::string, Function> local_env = find_transitive_calls(func);
        env.insert(local_env.begin(), local_env.end());
    }
    std::set<std::string> output_set;
    for (const auto &output : outputs) {
        output_set.insert(output.name());
    }
    std::vector<std::string> order = realization_order(output_functions, env).first;
    std::map<std::string, int> called_counts;
    std::map<std::string, int> call_counts;
    // Dependency analysis
    for (auto it = order.begin(); it != order.end(); it++) {
        Func func(env[*it]);
        int count = 0;
        std::map<std::string, int> calls = count_calls(func, count);
        called_counts.insert(calls.begin(), calls.end());
        call_counts[func.name()] = count;
    }
    // Traverse from the consumers to the producers
    for (auto it = order.rbegin(); it != order.rend(); it++) {
        Func func(env[*it]);
        if (skip_functions.find(func.name()) != skip_functions.end()) {
            continue;
        }
        int rvars_count = 0;
        if (rvars_count == 0 && call_counts[func.name()] <= 1 && output_set.find(func.name()) == output_set.end()) {
            // If the function isn't doing much and is not in the outputs, inline it
            func.compute_inline();
            continue;
        }

        Box bounds = func_bounds[*it];
        std::vector<int> int_bounds;
        for (int i = 0; i < (int)bounds.size(); i++) {
            Interval interval = bounds[i];
            Expr extent = simplify(interval.max - interval.min + 1);
            for (const auto &param : parameters) {
                extent = substitute(param.first, Expr(param.second), extent);
            }
            extent = simplify(extent);
            const int64_t *extent_int = as_const_int(extent);
            user_assert(extent_int != nullptr) << "extent:" << extent << " is not constant.\n";
            int_bounds.push_back(*extent_int);
        }

        func.compute_root();
        // initial definition is easy: everything is pure variables
        // just parallelize and vectorize if there are enough places to launch threads
        int tile_width = 16;
        int tile_height = 16;
        int min_gpu_threads = 128;
        int min_cpu_threads = 8;
        int min_threads = options.gpu ? min_gpu_threads : min_cpu_threads;
        int vectorize_width = 8;
        bool tilable = false;
        if ((int)int_bounds.size() >= 2 &&
                int_bounds[0] >= tile_width &&
                int_bounds[1] >= tile_height &&
                (int_bounds[0] / tile_width) * (int_bounds[1] / tile_height) >= min_threads) {
            Var xo, yo, xi, yi;
            Var tile_index;
            if (options.gpu) {
                func.gpu_tile(func.args()[0], func.args()[1], xo, yo, xi, yi, tile_width, tile_height);
            } else {
                func.tile(func.args()[0], func.args()[1], xo, yo, xi, yi, tile_width, tile_height)
                    .fuse(xo, yo, tile_index)
                    .parallel(tile_index)
                    .vectorize(xi, vectorize_width);
            }
            tilable = true;
        }

        for (int update_id = 0; update_id < func.num_update_definitions(); update_id++) {
            const std::vector<ReductionVariable> &rvars =
                func.update(update_id).get_schedule().rvars();
            // TODO: gracefully fallback if factorization is impossible
            if (!tilable && rvars.size() > 0) {
                std::vector<int> rvar_extents;
                Expr extent = rvars[0].extent;
                for (const auto &param : parameters) {
                    extent = substitute(param.first, Expr(param.second), extent);
                }
                extent = simplify(extent);
                const int64_t *extent_int = as_const_int(extent);
                user_assert(extent_int != nullptr) << "extent:" << extent << " is not constant.\n";
                rvar_extents.push_back(*extent_int);
                for (int arg_id = 1; arg_id < (int)rvars.size(); arg_id++) {
                    Expr extent = rvars[arg_id].extent;
                    for (const auto &param : parameters) {
                        extent = substitute(param.first, Expr(param.second), extent);
                    }
                    extent = simplify(extent);
                    const int64_t *extent_int = as_const_int(extent);
                    user_assert(extent_int != nullptr) << "extent:" << extent << " is not constant.\n";
                    rvar_extents.push_back(*extent_int);
                }
                // Tile rvar
                int dim_width = -1;
                int dim_height = -1;
                for (int rvar_id = 0; rvar_id < (int)rvars.size(); rvar_id++) {
                    if (dim_width == -1 && rvar_extents[rvar_id] >= tile_width) {
                        dim_width = rvar_id;
                    } else if (dim_height == -1 && rvar_extents[rvar_id] >= tile_height) {
                        assert(dim_width != -1);
                        dim_height = rvar_id;
                        break;
                    }
                }
                if (dim_width != -1 && dim_height != -1) {
                    if (options.gpu) {
                        // Each GPU thread covers 8 reductions over x
                        // Launch height number of threads per block
                        RVar rxo, rxi;
                        func.update(update_id)
                            .split(RVar(rvars[dim_width].var), rxo, rxi, tile_width);
                        Var xo, y;
                        Func interm = func.update(update_id)
                                          .rfactor({{rxo, xo},
                                                    {RVar(rvars[dim_height].var), y}});
                        std::vector<VarOrRVar> new_order;
                        new_order.push_back(rxi);
                        new_order.push_back(y);
                        new_order.push_back(xo);
                        for (const auto &arg : func.args()) {
                            new_order.push_back(arg);
                        }
                        interm.compute_root()
                              .reorder(y, xo)
                              .gpu_blocks(xo)
                              .gpu_threads(y);
                        interm.update()
                              .reorder(new_order)
                              .gpu_blocks(xo)
                              .gpu_threads(y);
                    } else {
                        // Parallel on tiles and vectorize inside tile
                        RVar rxo, ryo, rxi, ryi;
                        func.update(update_id)
                            .split(RVar(rvars[dim_width].var), rxo, rxi, tile_width)
                            .split(RVar(rvars[dim_height].var), ryo, ryi, tile_height);
                        Var xo, yo, xi;
                        Func interm = func.update(update_id)
                                          .rfactor({{rxo, xo},
                                                    {ryo, yo},
                                                    {rxi, xi}});
                        Var tile_index;
                        std::vector<VarOrRVar> new_order;
                        new_order.push_back(ryi);
                        new_order.push_back(xi);
                        for (const auto &arg : func.args()) {
                            new_order.push_back(arg);
                        }
                        new_order.push_back(tile_index);
                        interm.compute_root()
                              .fuse(xo, yo, tile_index)
                              .parallel(tile_index)
                              .vectorize(xi);
                        interm.update()
                              .fuse(xo, yo, tile_index)
                              .reorder(new_order)
                              .parallel(tile_index)
                              .vectorize(xi);
                    }
                }
            }
            std::vector<Var> pure_args;
            std::vector<Expr> update_args = func.update_args(update_id);
            std::vector<int> pure_arg_bounds;
            for (int arg_id = 0; arg_id < (int)update_args.size(); arg_id++) {
                Expr arg = update_args[arg_id];
                const Variable *var = arg.as<Variable>();
                if (var != nullptr &&
                        !var->param.defined() &&
                        !var->image.defined() &&
                        !var->reduction_domain.defined()) {
                    pure_args.push_back(Var(var->name));
                    pure_arg_bounds.push_back(int_bounds[arg_id]);
                }
            }

            if ((int)pure_arg_bounds.size() >= 2 &&
                    pure_arg_bounds[0] >= tile_width &&
                    pure_arg_bounds[1] >= tile_height &&
                    (int_bounds[0] / tile_width) * (int_bounds[1] / tile_height) >= min_threads) {
                Var xo, yo, xi, yi;
                Var tile_index;
                if (options.gpu) {
                    func.update(update_id)
                        .gpu_tile(pure_args[0], pure_args[1], xo, yo, xi, yi, tile_width, tile_height);
                } else {
                    func.update(update_id)
                        .tile(pure_args[0], pure_args[1], xo, yo, xi, yi, tile_width, tile_height)
                        .fuse(xo, yo, tile_index)
                        .parallel(tile_index)
                        .vectorize(xi, vectorize_width);
                }
            }
        }
    }
}

void simple_autoschedule(Func &output,
                         const std::map<std::string, int> &parameters,
                         const std::vector<std::pair<int, int>> &output_bounds,
                         const SimpleAutoscheduleOptions &options,
                         const std::set<std::string> &skip_functions) {
    std::vector<Func> outputs{output};
    std::vector<std::vector<std::pair<int, int>>> vector_output_bounds{output_bounds};
    return simple_autoschedule(outputs,
                               parameters,
                               vector_output_bounds,
                               options,
                               skip_functions);
}

void print_func(const Func &func, bool ignore_bc, bool ignore_non_adjoints, bool recursive, int depth) {
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
    if (depth >= 0) {
        lowest_index = (int)funcs.size() - 1 - depth;
    }

    for (int i = (int)funcs.size() - 1; i >= lowest_index; i--) {
        // TODO: "recursive" is a bit misleading here since there' no actual recursion of print_func
        if (!recursive && funcs[i].name() != func.name()) {
            continue;
        }
        const char *ce = "constant_exterior";
        const char *re = "repeat_edge";
        if (ignore_bc && (funcs[i].name().substr(0, strlen(ce)) == std::string(ce) ||
                funcs[i].name().substr(0, strlen(re)) == std::string(re) ||
		funcs[i].name().find("_ce") != std::string::npos)) {
            continue;
        }
        if (ignore_non_adjoints && funcs[i].name().find("_d_def__") == std::string::npos) {
            continue;
        }
        Func func = funcs[i];
        Internal::debug(0) << "  funcs[" << i << "]: " << func.name() << "\n";
        for (int update_id = -1; update_id < func.num_update_definitions(); update_id++) {
            Internal::ReductionDomain rdom;
            if (update_id >= 0) {
                Internal::debug(0) << "    update:" << func.name() << "(";
                if (func.update_args(update_id).size() > 0) {
                    Internal::debug(0) << Internal::simplify(func.update_args(update_id)[0]);
                    for (int i = 1; i < (int)func.update_args(update_id).size(); i++) {
                        Internal::debug(0) << ", " << 
                            Internal::simplify(func.update_args(update_id)[i]);
                    }
                }
                Internal::debug(0) << ") =";
                auto vals = func.update_values(update_id).as_vector();
                for (auto val : vals) {
                    Internal::debug(0) << " " << Internal::simplify(val);
                }
                Internal::debug(0) << "\n";
                rdom = Internal::extract_rdom(Internal::simplify(func.update_value(update_id)));
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
                    Internal::debug(0) << " " << Internal::simplify(val);
                }
                Internal::debug(0) << "\n";
                rdom = Internal::extract_rdom(Internal::simplify(func.value()));
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
    Func f_input("f_input");
    f_input(x) = input(x);
    Func blur("blur");
    blur(x) = f_input(x) + f_input(x + 1);
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
    Buffer<float> d_clamped_buf = adjoints[FuncKey{f_input.name(), -1}].realize(1);
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
    Func f_input("f_input");
    f_input(x) = input(x);
    Func f_output("output");
    f_output(x, y) = f_input(y);

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

    Buffer<float> d_input = adjoints[FuncKey{f_input.name(), -1}].realize(2);
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
    f_output(2) = f_input(1);

    Func f_loss("f_loss");
    RDom r(0, 3);
    f_loss() = 0.f;
    f_loss() += f_output(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> d_input = adjoints[FuncKey{f_input.name(), -1}].realize(3);
    CMP(d_input(0), 1.0f);
    CMP(d_input(1), 1.0f);
    CMP(d_input(2), 0.0f);
}

void test_rdom_update() {
    Var x("x");
    float input_data[] = {1.0f, 2.0f, 3.0f};
    Buffer<float> input(input_data, 3, "input");
    Func f_input("f_input");
    f_input(x) = input(x);
    Func f_output("f_output");
    RDom r0(1, 2), r1(3, 4);
    f_output(x) = f_input(x);
    f_output(r0) = f_input(r0 - 1);
    f_output(r1) = 0.f;

    Func f_loss("f_loss");
    RDom r_target(0, 5);
    f_loss() = 0.f;
    f_loss() += f_output(r_target);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> d_input = adjoints[FuncKey{f_input.name(), -1}].realize(3);
    CMP(d_input(0), 2.0f);
    CMP(d_input(1), 1.0f);
    CMP(d_input(2), 0.0f);
}

void test_repeat_edge() {
    Var x("x");
    float input_data[] = {1.f, 2.f};
    Buffer<float> input(input_data, 2, "input");
    Func f_input("f_input");
    f_input(x) = input(x);
    Func clamped = BoundaryConditions::repeat_edge(f_input, {{0, 2}});
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
    Buffer<float> d_input_buf = adjoints[FuncKey{f_input.name(), -1}].realize(2);
    // d loss / d i0 = 1
    // d loss / d i1 = 5
    CMP(d_input_buf(0), 1.f);
    CMP(d_input_buf(1), 5.f);
}

void test_second_order() {
    Var x("x");
    float input_data[] = {1.f};
    Buffer<float> input(input_data, 1, "input");
    Func f_input("f_input");
    f_input(x) = input(x);
    Func polynomial("polynomial");
    // x^2 + 3x + 4.f
    polynomial(x) = f_input(x) * f_input(x) + 3.f * f_input(x) + 4.f;
    Derivative d = propagate_adjoints(polynomial);
    Func d_input = d.adjoints[FuncKey{f_input.name(), -1}];
    Derivative d2 = propagate_adjoints(d_input);
    Func d2_input = d2.adjoints[FuncKey{f_input.name(), -1}];

    Buffer<float> buf = d_input.realize(1);
    Buffer<float> buf2 = d2_input.realize(1);
    // d/dx = 2x + 3
    CMP(buf(0), 5.f);

    // d^2/dx^2 = 2
    CMP(buf2(0), 2.f);
}

void test_implicit_vars() {
    Var x("x");
    float input_data[] = {1.f, 2.f};
    Buffer<float> input(input_data, 2, "input");
    Func f_input("f_input");
    f_input(x) = input(x);
    Func copy("copy");
    copy(_) = f_input(_);
    RDom r(0, 2);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += copy(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> d_input_buf = adjoints[FuncKey{f_input.name(), -1}].realize(2);
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
    Func f_input("f_input");
    f_input(x) = input(x);
    Func tuple("tuple");
    tuple(x) = Tuple(f_input(x), f_input(x + 1));
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

    Buffer<float> d_input_buf = adjoints[FuncKey{f_input.name(), -1}].realize(3);
    CMP(d_input_buf(0), 1.f);
    CMP(d_input_buf(1), 2.f);
    CMP(d_input_buf(2), 1.f);
}

void test_floor_ceil() {
    Var x("x");
    float input_data[] = {1.f, 2.f, 3.f};
    Buffer<float> input(input_data, 3, "input");
    Func f_input("f_input");
    f_input(x) = input(x);
    Func floor_output("floor_output");
    floor_output(x) = f_input(cast<int>(floor(x / 4.f)));
    Func ceil_output("ceil_output");
    ceil_output(x) = f_input(cast<int>(ceil(x / 4.f)));
    Func output("output");
    output(x) = ceil_output(x) + floor_output(x);
    RDom r(0, 8);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += output(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;
    // floor_output(0~3) == input[0]
    // floor_output(4~7) == input[1]
    // ceil_output(0) == input[0]
    // ceil_output(1~4) == input[1]
    // ceil_output(5~7) = input[2]

    Buffer<float> d_input_buf = adjoints[FuncKey{f_input.name(), -1}].realize(3);

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
    Func f_input("f_input");
    f_input(x) = input(x);
    Func output("output");
    RDom r(0, 4);
    output(x) = 0.f;
    output(x) += f_input(4 * x + r);
    RDom r_loss(0, 2);
    Func f_loss("f_loss");
    f_loss() = 0.f;
    f_loss() += output(r_loss);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;
    // output(0) = \sum input(0~4)
    // output(1) = \sum input(5~8)
    Buffer<float> d_input_buf = adjoints[FuncKey{f_input.name(), -1}].realize(10);

    for (int i = 0; i < 8; i++) {
        CMP(d_input_buf(i), 1.f);
    }
    CMP(d_input_buf(8), 0.f);
    CMP(d_input_buf(9), 0.f);
}

#if 0
void test_ATA() {
    Var x("x");
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    Buffer<float> input(5, "input");
    for (int i = 0; i < 5; i++) {
        input(i) = dist(rng);
    }
    Func clamped("clamped");
    Expr clamped_x = Halide::clamp(x, 0, input.width() - 1);
    clamped(x) = input(clamped_x);
    Buffer<float> kernel(3, "kernel");
    for (int i = 0; i < 3; i++) {
        kernel(i) = dist(rng);
    }
    Func kernel_func("kernel_func");
    kernel_func(x) = kernel(x);
    Func Ax("Ax");
    RDom support(0, 3);
    Ax(x) = 0.f;
    Ax(x) += clamped(x + support.x - 1) * kernel_func(support.x);
    Func ATAx("ATAx");
    ATAx(x) = 0.f;
    ATAx(x) += Ax(x + support.x - 1) * kernel_func(2 - support.x);

    Buffer<float> adjoints(5);
    for (int i = 0; i < 5; i++) {
        adjoints(i) = dist(rng);
    }

    Derivative d = propagate_adjoints(ATAx, adjoints);

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
#endif

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
    test_implicit_vars();
    test_tuple();
    test_floor_ceil();
    test_downsampling();
    debug(0) << "Derivative test passed\n";
}

} // namespace Internal
} // namespace Halide
