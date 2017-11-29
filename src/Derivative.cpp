#include "Derivative.h"

#include "DerivativeUtils.h"
#include "BoundaryConditions.h"
#include "Simplify.h"
#include "Solve.h"
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

    std::map<FuncKey, RDom> get_reductions() const {
        return reductions;
    };

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
    std::map<FuncKey, RDom> reductions;
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
    std::vector<std::string> order = realization_order({output.function()}, env);
    std::vector<Func> funcs;
    funcs.reserve(order.size());
    Internal::debug(0) << "Sorted Func list:" << "\n";
    for (const auto &func_name : order) {
        Internal::debug(0) << "  . " << func_name << "\n";
    }
    for (const auto &func_name : order) {
        Func func(env[func_name]);
        // Avoid implicit variables
        // TODO: FIX THIS
        if (func.args().size() > 0 && func.args()[0].is_implicit()) {
            continue;
        }
        funcs.push_back(Func(env[func_name]));
    }

    internal_assert(funcs.size() > 0);

    debug(0) << "ReverseAccumulationVisitor: infering bounds...";
    func_bounds = inference_bounds(output, output_bounds);
    debug(0) << "done\n";

    // Create a stub for each function to accumulate adjoints
    for (int func_id = 0; func_id < (int)funcs.size(); func_id++) {
        const Func &func = funcs[func_id];
        for (int update_id = -1; update_id < func.num_update_definitions(); update_id++) {
            Func adjoint_func(func.name() + "_" + std::to_string(update_id + 1) + "_d_def__");
            bool is_last = func_id == (int)funcs.size() - 1 &&
                           update_id == func.num_update_definitions() - 1;
            if (is_last) {
                adjoint_func(func.args()) = adjoint(func.args());
            } else {
                adjoint_func(func.args()) = 0.f;
            }
            FuncKey func_key{func.name(), update_id};
            adjoint_funcs[func_key] = adjoint_func;
        }
    }

    // Traverse functions from back to front for reverse accumulation
    for (int func_id = funcs.size() - 1; func_id >=  0; func_id--) {
        const Func &func = funcs[func_id];
        current_func = func;

        // Traverse from the last update to first
        for (int update_id = func.num_update_definitions() - 1;
                update_id >= -1; update_id--) {
            current_update_id = update_id;
            FuncKey func_key{func.name(), update_id};
            internal_assert(func_bounds.find(func.name()) != func_bounds.end());

            // Set up boundary condition
            const Box &bounds = func_bounds[func.name()];
            // Apply boundary condition if this is the first visit
            if (update_id == func.num_update_definitions() - 1) {
                Func adjoint_func = adjoint_funcs[func_key];
                adjoint_func = BoundaryConditions::constant_exterior(
                    adjoint_func, 0.f, box_to_vector(bounds));
                // Give it a better name
                Func tmp(func.name() + "_" + std::to_string(update_id + 1) + "_d__");
                tmp(adjoint_func.args()) = adjoint_func(adjoint_func.args());
                adjoint_funcs[func_key] = tmp;
            }

            // Propagate the adjoints to next update
            // Example:
            // f(x) = ...
            // f(1) = ...
            // Need to propagate back to all x while masking 1
            if (update_id >= 0) {
                FuncKey next_func_key{func.name(), update_id - 1};
                Func next_adjoint_func = adjoint_funcs[next_func_key];
                // TODO: create new functions if left hand side is pure
                next_adjoint_func(next_adjoint_func.args()) =
                    adjoint_funcs[func_key](next_adjoint_func.args());
                next_adjoint_func(func.update_args(update_id)) = 0.f;
            }

            // Topologically sort the expressions
            std::vector<Expr> expr_list =
                update_id >= 0 ? sort_expressions(func.update_value(update_id)) :
                                 sort_expressions(func.value());

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

            // Retrieve previously propagated adjoint for the Func, apply it to expression adjoints
            std::vector<Expr> update_args;
            if (update_id >= 0) {
                update_args = func.update_args(update_id);
            } else {
                update_args.reserve(func.args().size());
                for (const auto &var : func.args()) {
                    update_args.push_back(var);
                }
            }
            expr_adjoints[(const BaseExprNode *)expr_list.back().get()] =
                Call::make(adjoint_funcs[func_key].function(), update_args);

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
    if (op->name == "exp_f32") {
        // d/dx exp(x) = exp(x)
        for (size_t i = 0; i < op->args.size(); i++) {
            accumulate(op->args[i], adjoint * exp(op->args[i]));
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
    }

    // Halide function call
    if (op->func.defined()) {
        Function func(op->func);
        // Avoid implicit functions
        // TODO: FIX THIS
        if (func.args().size() > 0 && Var::is_implicit(func.args()[0])) {
            return;
        }

        // We are scattering to this function
        // debug(0) << "Scattering to " << func.name() << "\n";

        // TODO: check if we need this elsewhere
        // Add Let expressions
        adjoint = add_let_expression(adjoint, let_var_mapping, let_variables);
        std::vector<Expr> lhs = op->args;
        for (int i = 0; i < (int)lhs.size(); i++) {
            lhs[i] = add_let_expression(lhs[i], let_var_mapping, let_variables);
        }

        // debug(0) << "lhs is:";
        // for (const auto &arg : lhs) {
        //     debug(0) << " " << arg;
        // }
        // debug(0) << "\n";
        // debug(0) << "adjoint is:" << simplify(adjoint) << "\n";

        // If target is the current function itself, send to previous update
        FuncKey func_key = func.name() != current_func.name() ?
                           FuncKey{func.name(), func.updates().size() - 1} :
                           FuncKey{func.name(), current_update_id - 1};
        assert(adjoint_funcs.find(func_key) != adjoint_funcs.end());
        Func& func_to_update = adjoint_funcs[func_key];

        // Gather argument & bounds information
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
        // we canonicalize by first try to subsitute with pure variables
        // if that fails we will replace variables on lhs with RDoms

        // Try to do the canonicalization
        // Create a set of new substitution variables
        std::vector<Var> new_args;
        std::set<std::string> new_args_set;
        for (int i = 0; i < (int)func_to_update.args().size(); i++) {
            new_args.push_back(Var("u" + std::to_string(i) + "_"));
            new_args_set.insert(new_args.back().name());
        }

        std::vector<bool> canonicalized(lhs.size(), false);
        // We canonicalize the left hand side arguments (op->args) so that it's always x, y, z, ...
        // First gather all arguments that contains multiple pure variables
        // we don't want to mess with system solving yet, so we invalidate all of them
        // 
        // Given:
        // g(x, y, z) = f(x, y-1, z+1)
        // we get:
        // d_f(x, y - 1, z + 1) += d_g(x,y,z)
        // Goal: rewrite to
        //  ==> d_f(x,y,z) += d_g(x,y+1,z-1)
        // 
        // This is currently simple and conservative, solving each dimension independently, so 
        // inter-dependencies like:
        // g(x, y) = f(x*y, x+y)
        // can't be simplified. In principle this can be inverted by solving a system of equations.
        std::set<std::string> invalidated_variables;
        for (int i = 0; i < (int)lhs.size(); i++) {
            std::vector<std::string> variables =
                gather_variables(lhs[i], vars_to_strings(current_args));
            // skip cases where there are cross-dimension dependencies
            if (variables.size() != 1) {
                for (const auto &var : variables) {
                    invalidated_variables.insert(var);
                }
                continue;
            }
        }
        for (int i = 0; i < (int)lhs.size(); i++) {
            // Gather all pure variables at op->args[i], substitute them with new_args
            // For now only support single pure variable
            std::vector<std::string> variables =
                gather_variables(lhs[i], vars_to_strings(current_args));
            if (variables.size() != 1) {
                continue;
            }

            // If it appeared in other argument we should also fail
            if (invalidated_variables.find(variables[0]) != invalidated_variables.end()) {
                continue;
            }

            // Let new_args[i] == op->args[i]
            // e.g. u_1 == y - 1 in the above example
            SolverResult result = solve_expression(new_args[i] == lhs[i], variables[0]);
            // debug(0) << "solving " << new_args[i] << " " << lhs[i] << " for " << variables[0] << "\n";
            if (!result.fully_solved) {
                // debug(0) << "expression not fully solved" << "\n";
                continue;
            }

            Expr result_rhs = result.result;
            if (result.result.as<Let>() != nullptr) {
                const Let *let_expr = result.result.as<Let>();
                // debug(0) << "we have a let " << result.result << "\n";
                // debug(0) << let_expr->value << " " << let_expr->body << "\n";
                result_rhs = substitute(let_expr->name, let_expr->value, let_expr->body);
                if (result_rhs.as<And>() != nullptr) {
                    // TODO(mgharbi): this is quite dirty and brittle, what's the right solution?
                    const And *and_expr = result_rhs.as<And>();
                    // debug(0) << "we have an And clause " << and_expr << "\n";
                    result_rhs = and_expr->a;
                }
            } else {
                result_rhs = result.result;
            }

            // y == u_1 + 1
            if (result_rhs.as<EQ>() != nullptr) {
                // Checking whether the lhs is a single variable
                Expr result_lhs = result_rhs.as<EQ>()->a;
                const Variable *lhs_var = result_lhs.as<Variable>();
                if (lhs_var == nullptr) {
                    // debug(0) << "expression not fully solved";
                    continue;
                }

                internal_assert(lhs_var->name == variables[0]);
                result_rhs = result_rhs.as<EQ>()->b;
            } else {
                internal_error << "coult not solve expression\n";
            }
            // debug(0) << "result : " << result_rhs << "\n";

            // Replace pure variable with the reverse
            adjoint = substitute(variables[0], result_rhs, adjoint);

            lhs[i] = func_to_update.args()[i];
            canonicalized[i] = true;
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
                std::vector<std::string> variables = gather_variables(lhs_arg, func.args());
                RDom r(bounds);
                for (int var_id = 0; var_id < (int)variables.size(); var_id++) {
                    for (int arg_id = 0; arg_id < (int)current_args.size(); arg_id++) {
                        if (current_args[arg_id].name() == variables[var_id]) {
                            lhs[lhs_id] = substitute(variables[var_id], r_bounds[arg_id], lhs[lhs_id]);
                            adjoint = substitute(variables[var_id], r_bounds[arg_id], adjoint);
                            break;
                        }
                    }
                }
            }
        }

        // For each free variable on the rhs, replace it with current bounds
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
        // For each variable in lhs, check if it is a single rvar
        // if so we can rewrite it back to pure variables
        // TODO: is this the best way?
        for (int i = 0; i < (int)lhs.size(); i++) {
            auto &lhs_arg = lhs[i];
            const Variable *var = lhs_arg.as<Variable>();
            if (var != nullptr) {
                if (var->reduction_domain.defined()) {
                    lhs_arg = substitute(var->name, func_to_update_args[i], lhs_arg);
                    adjoint = substitute(var->name, func_to_update_args[i], adjoint);
                }
            }
        }

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
        if (merged_bounds.size() > 0) {
            RDom r(merged_bounds);
            for (int i = 0; i < r.dimensions(); i++) {
                adjoint = substitute(var_names[i], r[i], adjoint);
                for (auto &lhs_arg : lhs) {
                    lhs_arg = substitute(var_names[i], r[i], lhs_arg);
                }
            }
            reductions[FuncKey{func_to_update.name(), func_to_update.num_update_definitions()}] = r;
        } else {
            reductions[FuncKey{func_to_update.name(), func_to_update.num_update_definitions()}] = RDom();
        }

        for (int i = 0; i < (int)func_to_update_args.size(); i++) {
            // substitute the new_args back to original variables
            adjoint = substitute(new_args[i].name(), func_to_update_args[i], adjoint);
        }

        func_to_update(lhs) += adjoint;

        // debug(0) << "lhs after canonicalization:";
        // for (const auto &arg : lhs) {
        //     debug(0) << " " << arg;
        // }
        // debug(0) << "\n";
        // debug(0) << "adjoint after canonicalization:" << simplify(adjoint) << "\n";
        // print_func(func_to_update);
    }
}

} // namespace Internal


Derivative propagate_adjoints(const Func &output,
                              const Func &adjoint,
                              const std::vector<std::pair<Expr, Expr>> &output_bounds) {
    user_assert(output.dimensions() == adjoint.dimensions());
    Internal::ReverseAccumulationVisitor visitor;
    visitor.propagate_adjoints(output, adjoint, output_bounds);
    return Derivative{visitor.get_adjoint_funcs(),
                      visitor.get_reductions()};
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

void print_func(const Func &func) {
    // Internal::debug(0) << "Printing function:" << func.name() << "\n";
    // Topologically sort the functions
    std::map<std::string, Internal::Function> env =
        find_transitive_calls(func.function());
    std::vector<std::string> order = realization_order({func.function()}, env);
    std::vector<Func> funcs;
    funcs.reserve(order.size());
    for (const auto &func_name : order) {
        Func func(env[func_name]);
        // Avoid implicit functions
        // TODO: FIX THIS
        if (func.args().size() > 0 && func.args()[0].is_implicit()) {
            continue;
        }
        funcs.push_back(func);
    }

    for (int i = (int)funcs.size() - 1; i >= 0; i--) {
        Func &func = funcs[i];
        // Internal::debug(0) << "  funcs[" << i << "]: " << func.name() << "\n";
        for (int update_id = -1; update_id < func.num_update_definitions(); update_id++) {
            Internal::ReductionDomain rdom;
            if (update_id >= 0) {
                // Internal::debug(0) << "    update:" << func.name() << "(" <<
                    // Internal::simplify(func.update_args(update_id)[0]);
                // for (int i = 1; i < (int)func.update_args(update_id).size(); i++) {
                //     Internal::debug(0) << ", " << Internal::simplify(func.update_args()[i]);
                // }
                // Internal::debug(0) << ") = " << Internal::simplify(func.update_value(update_id)) << "\n";
                rdom = Internal::extract_rdom(Internal::simplify(func.update_value(update_id)));
            } else {
                // Internal::debug(0) << "    " << func.name() << "(" << func.args()[0];
                // for (int i = 1; i < (int)func.args().size(); i++) {
                //     Internal::debug(0) << ", " << Internal::simplify(func.args()[i]);
                // }
                // Internal::debug(0) << ") = " << Internal::simplify(func.value()) << "\n";
                rdom = Internal::extract_rdom(Internal::simplify(func.value()));
            }

            if (rdom.defined()) {
                // Internal::debug(0) << "    RDom:";
                // for (int i = 0; i < (int)rdom.domain().size(); i++) {
                //     Internal::debug(0) << " (" <<
                //         Internal::simplify(rdom.domain()[i].min) << ", " <<
                //         Internal::simplify(rdom.domain()[i].extent) << ")";
                // }
                // Internal::debug(0) << "\n";
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

void test_scalar() {
    Func x("x");
    x() = 5.f;
    Func y("y");
    y() = x() * x() + 2.f * x() + 5.f;
    Derivative d = propagate_adjoints(y);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> dydx = adjoints[FuncKey{x.name(), -1}].realize();
    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

    // dydx = 2x + 2 = 12
    CMP(dydx(0), 12.f);
#undef CMP
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
    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

    CMP(d_blur_buf(0), 2 * blur_buf(0));
    CMP(d_blur_buf(1), 2 * blur_buf(1));
    Buffer<float> d_clamped_buf = adjoints[FuncKey{clamped.name(), -1}].realize(2);
    CMP(d_clamped_buf(0), d_blur_buf(0));
    CMP(d_clamped_buf(1), d_blur_buf(0) + d_blur_buf(1));

#undef CMP
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
    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

    CMP(d_blur_buf(0), 2 * blur_buf(0));
    Buffer<float> d_clamped_buf = adjoints[FuncKey{f_input.name(), -1}].realize(1);
    CMP(d_clamped_buf(0), d_blur_buf(0));

#undef CMP
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
    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

    CMP(d_blur_buf(0), 2 * blur_buf(0));
    CMP(d_blur_buf(1), 2 * blur_buf(1));
    Buffer<float> d_clamped_buf = adjoints[FuncKey{clamped.name(), -1}].realize(2);
    CMP(d_clamped_buf(0), d_blur_buf(0));
    CMP(d_clamped_buf(1), d_blur_buf(0) + d_blur_buf(1));

#undef CMP
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
    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

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

#undef CMP
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
    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

    Buffer<float> d_output = adjoints[FuncKey{f_output.name(), -1}].realize(2, 2);
    CMP(d_output(0, 0), 2);
    CMP(d_output(1, 0), 2);
    CMP(d_output(0, 1), 4);
    CMP(d_output(1, 1), 4);

    Buffer<float> d_input = adjoints[FuncKey{f_input.name(), -1}].realize(2);
    CMP(d_input(0), 4);
    CMP(d_input(1), 8);

#undef CMP
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
    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

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
#undef CMP
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
    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

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
#undef CMP
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

    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

    Buffer<float> d_input = adjoints[FuncKey{f_input.name(), -1}].realize(3);
    CMP(d_input(0), 1.0f);
    CMP(d_input(1), 1.0f);
    CMP(d_input(2), 0.0f);
#undef CMP
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

    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

    Buffer<float> d_input = adjoints[FuncKey{f_input.name(), -1}].realize(3);
    CMP(d_input(0), 2.0f);
    CMP(d_input(1), 1.0f);
    CMP(d_input(2), 0.0f);
#undef CMP
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
    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

    // d loss / d i0 = 1
    // d loss / d i1 = 5
    CMP(d_input_buf(0), 1.f);
    CMP(d_input_buf(1), 5.f);

#undef CMP
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

    const float eps = 1e-6;
#define CMP(x, target) \
    internal_assert(fabs((x) - (target)) < eps) << \
        "Expected " << (target) << " instead of " << (x) << "\n";

    // d/dx = 2x + 3
    CMP(buf(0), 5.f);

    // d^2/dx^2 = 2
    CMP(buf2(0), 2.f);
#undef CMP
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
    debug(0) << "Derivative test passed\n";
}

} // namespace Internal
} // namespace Halide
