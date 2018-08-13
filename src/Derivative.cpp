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
#include "Associativity.h"

#include <iostream>
#include <cmath>

namespace std {
inline Halide::float16_t fabs(Halide::float16_t x) {
    return x > Halide::float16_t(0) ? x : -x;
}
}

namespace Halide {
namespace Internal {

bool check_opname(const std::string &op_name,
                  const std::string &func_name) {
    return op_name == (func_name + "_f16") ||
           op_name == (func_name + "_f32") ||
           op_name == (func_name + "_f64");
};

/** Compute derivatives through reverse accumulation
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

    // For each expression, we store the accumulated adjoints expression
    std::map<const BaseExprNode *, Expr> expr_adjoints;
    // For each function and each update, we store the accumulated adjoints func
    std::map<FuncKey, Func> adjoint_funcs;
    // Let variables and their mapping
    std::map<std::string, Expr> let_var_mapping;
    std::vector<std::string> let_variables;
    // Bounds of functions
    std::map<std::string, Box> func_bounds;
    // Current function that scatters its adjoints to its dependencies
    Func current_func;
    // Current update of the function
    int current_update_id;
    // Are we using a reduction domain for keeping track of intermediate values
    bool current_has_intermediate_rdom;
    // If a function is self referencing we accumulate the
    // adjoints to this function.
    // We propagate this temporary adjoint to proper place later
    Func tmp_adjoint_func;
    // We compute the derivatives in several passes.
    // Sometimes we don't want to propagate through Halide function calls
    bool is_forward_canonicalizing_phase;
    bool is_self_referencing_phase;
};

void ReverseAccumulationVisitor::propagate_adjoints(
        const Func &output,
        const Func &adjoint,
        const std::vector<std::pair<Expr, Expr>> &output_bounds) {
    // Topologically sort the functions
    std::map<std::string, Function> env = find_transitive_calls(output.function());
    std::vector<std::string> order =
        realization_order({output.function()}, env).first;
    std::vector<Func> org_funcs;
    org_funcs.reserve(order.size());
    // Internal::debug(0) << "Sorted Func list:" << "\n";
    // for (const auto &func_name : order) {
    //     Internal::debug(0) << "  . " << func_name << "\n";
    // }
    for (const auto &func_name : order) {
        Func func(env[func_name]);
        org_funcs.push_back(Func(env[func_name]));
    }
    internal_assert(org_funcs.size() > 0);

    std::vector<Func> funcs;
    funcs.reserve(org_funcs.size());
    std::vector<std::vector<Expr>> intermediate_version_number;
    intermediate_version_number.reserve(org_funcs.size());
    // Preprocess the forward functions and store them into funcs
    //
    // Reverse accumulation with self-update is a bit tricky.
    // The following cases, where the derivatives depend on intermediate values,
    // are troublesome for our adjoint propagation algorithm:
    // 1. The derivatives depend on values of previous updates
    // 2. The derivatives depend on values of current updates in a reduction domain
    // For example:
    //
    // f(x) = g(x)
    // f(x) = f(x) * f(x)
    //
    // f(x) = 0
    // f(x) = 2 * f(x) + g(r.x)
    //
    // f(x) = g(x)
    // f(r.x) = f(r.x) * f(r.x) + g(r.x)
    // f(x) = f(x) * f(x)
    //
    // The issue is that the self reference to f(x) makes propagation to g(x)
    // receives the wrong adjoints.
    // Our solution is to declare different intermediate versions of f.
    // If we need one more version of f, we add an extra dimension to f,
    // and the intermediate values is stored in the extra dimension.
    // We rewrite the three cases above to the following:
    //
    // f_(x, 0) = g(x)
    // f_(x, 1) = f_(x, 0) * f_(x, 0)
    // f(x) = f_(x, 1)
    //
    // f_(x, 0) = 0
    // f_(x, r.x + 1) = 2 * f_(x, r.x) + g(r.x)
    // f(x) = f_(x, r.x.max() + 1)
    //
    // f_(x, 0) = g(x)
    // f_(r.x, r.x + 1) = f_(r.x, r.x) * f_(r.x, r.x) + g(r.x)
    // f_(x, r.x.max() + 2) = f_(x, r.x.max() + 1) * f_(x, r.x.max() + 1)
    // f(x) = f_(x, r.x.max() + 2)
    //
    // The corresponding adjoints are then:
    //
    // d_f_(x, 1) += d_f(x)
    // d_f_(x, 0) += 2 * d_f_(x, 1) * f_(x, 0)
    // d_g(x) += d_f_(x, 0)
    //
    // d_f_(x, r.x + 1) += d_f(x)
    // d_f_(x, r.x) += 2 * d_f_(x, r.x + 1)
    // d_g(r.x) += d_f_(x, r.x + 1)
    //
    // d_f_(x, r.x + 1) += d_f(x)
    // d_f_(x, r.x) += 2 * d_f_(x, r.x + 1) * f(x, r.x)
    // d_g(r.x) += d_f(x, r.x + 1)
    // d_g(x) += d_f(x, 0)
    //
    // TODO: This does not work for reduction domains with predicates yet.
    //       The versioning trick assume the reduction domain is dense.
    //       A possible remedy is to create an extra index set which stores
    //       all visited reduction indices.
    //       For now we'll just throw an error if we detect predicates.
    // Setup flags for member functions
    is_forward_canonicalizing_phase = true;
    for (int func_id = 0; func_id < (int)org_funcs.size(); func_id++) {
        const Func &func = org_funcs[func_id];
        // Number of versions we need to declare for func
        Expr num_checkpoints = 1;
        std::vector<Expr> num_checkpoints_list;
        num_checkpoints_list.reserve(func.num_update_definitions());
        // +1 for the initial undef of the new function
        std::vector<Expr> version_number(
            func.num_update_definitions() + 1, Expr());
        for (int update_id = 0;
                update_id < func.num_update_definitions(); update_id++) {
            // Check case 2 first: is there a non-commutative reduction variable?
            const std::vector<Expr> &update_args = func.update_args(update_id);
            Tuple rhs_tuple = func.update_values(update_id);
            AssociativeOp associative_op =
                prove_associativity(func.name(),
                    update_args, rhs_tuple.as_vector());
            bool is_commutative = associative_op.commutative();
            if (associative_op.xs.size() == 0 ||
                    (associative_op.xs.size() == 1 && associative_op.xs[0].var == "")) {
                // Unary operation is also fine
                is_commutative = true;
            }
            bool is_associative = associative_op.associative();

            bool has_self_reference = false;
            if (!is_commutative || !is_associative) {
                // Check case 1: if the derivatives reference to 
                // previous self in an update
                // Take the derivative at expression level, the results are
                // stored in expr_adjoints
                std::vector<Expr> expr_list;
                Tuple rhs_tuple =
                    update_id < 0 ? func.values() : func.update_values(update_id);
                std::vector<const BaseExprNode *> output_exprs;
                const std::vector<Expr> &rhs_tuple_vector = rhs_tuple.as_vector();
                for (const auto &expr : rhs_tuple_vector) {
                    std::vector<Expr> value_expr_list = sort_expressions(expr);
                    expr_list.insert(expr_list.end(),
                        value_expr_list.begin(), value_expr_list.end());
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

                // Set the output adjoint to 1
                // We're not really propagating adjoints, just checking if there's
                // self references
                for (int i = 0; i < (int)output_exprs.size(); i++) {
                    expr_adjoints[output_exprs[i]] = 1.f;
                }

                // Traverse the expressions in reverse order
                for (auto it = expr_list.rbegin(); it != expr_list.rend(); it++) {
                    it->accept(this);
                }

                // For each adjoint expression, check if it references to the function
                for (const auto &it : expr_adjoints) {
                    Expr expr = it.second;
                    if (is_calling_function(func.name(), expr, let_var_mapping)) {
                        has_self_reference = true;
                        break;
                    }
                }
                expr_adjoints.clear();
            }

            if (has_self_reference || !is_commutative || !is_associative) {
                // Count the number of versions we need to store
                // Gather reduction domain
                std::map<std::string, ReductionVariableInfo> rvars =
                    gather_rvariables(func.update_values(update_id));
                Expr new_chkpts = 1;
                for (const auto &it : rvars) {
                    const ReductionVariableInfo &info = it.second;
                    if (info.domain.predicate().defined() &&
                            !is_const(info.domain.predicate())) {
                        user_error 
                            << "Detect predicates when computing derivatives in non commutative or"
                            << "non associative reduction domain. This is unsupported yet.";
                    }
                    new_chkpts *= info.extent;
                }
                if (rvars.size() > 0) {
                    version_number[update_id + 1] = num_checkpoints;
                }
                new_chkpts = simplify(new_chkpts);
                num_checkpoints_list.push_back(new_chkpts);
                num_checkpoints = simplify(num_checkpoints + new_chkpts);
            } else {
                num_checkpoints_list.push_back(0);
            }
        }
        if (is_const(num_checkpoints, 1)) {
            funcs.push_back(func);
            intermediate_version_number.push_back(
                std::vector<Expr>(func.num_update_definitions(), Expr()));
            continue;
        }
        // If we need more than one version of func, create a new function
        // with an extra dimension
        Func new_func(func.name() + "_chkpt");
        std::vector<Expr> args;
        args.reserve(func.args().size() + 1);
        for (const Var &var : func.args()) {
            args.push_back(var);
        }
        Var v("ver"); // version
        args.push_back(v);
        // Initialize with undef
        if (func.outputs() == 1) {
            new_func(args) = undef(func.value().type());
        } else {
            std::vector<Expr> undefs;
            undefs.reserve(func.values().size());
            const std::vector<Expr> &values = func.values().as_vector();
            for (const Expr &e : values) {
                undefs.push_back(undef(e.type()));
            }
        }
        // f_(x, 0) = f(x)
        args.back() = 0;
        if (func.outputs() == 1) {
            new_func(args) = func.value();
        } else {
            new_func(args) = func.values();
        }
        Expr current_chkpt = 0;
        for (int update_id = 0;
                update_id < func.num_update_definitions(); update_id++) {
            // Update checkpoint position
            Expr new_chkpt =
                simplify(current_chkpt + num_checkpoints_list[update_id]);
            Expr rhs_extra_arg;
            if (is_const(num_checkpoints_list[update_id], 0) ||
                    is_const(num_checkpoints_list[update_id], 1)) {
                args.back() = new_chkpt;
                rhs_extra_arg = current_chkpt;
            } else {
                // Loop over the versions of rdoms
                // f(x, r.x + 1) = f(x, r.x)
                // First we need to linearize the reduction domain
                std::map<std::string, ReductionVariableInfo> rvars_info =
                    gather_rvariables(func.update_values(update_id));
                std::vector<RVar> rvars = func.rvars(update_id);
                Expr index = current_chkpt;
                Expr stride = 1;
                for (int i = 0; i < (int)rvars.size(); i++) {
                    auto it = rvars_info.find(rvars[i].name());
                    internal_assert(it != rvars_info.end());
                    const ReductionVariableInfo &info = it->second;
                    RVar r(info.domain, info.index);
                    index += stride * (r - info.min);
                    stride *= info.extent;
                }
                args.back() = simplify(index + 1);
                rhs_extra_arg = index;
            }
            std::vector<Expr> update_values =
                func.update_values(update_id).as_vector();
            for (Expr &expr : update_values) {
                // Replace the call to original function with the new one
                expr = replace_call_add_argument(
                    func.name(), new_func, expr, rhs_extra_arg);
            }
            if (func.outputs() == 1) {
                new_func(args) = update_values[0];
            } else {
                new_func(args) = Tuple(update_values);
            }
            current_chkpt = new_chkpt;
        }
        // f(x) = f_(x, n)
        func.function().remove_updates();
        args.back() = current_chkpt;
        for (int i = 0; i < func.outputs(); i++) {
            func.function().definition().values()[i] = 
                Call::make(new_func.function(), args, i);
        }
        // Push both functions into list
        funcs.push_back(new_func);
        funcs.push_back(func);
        intermediate_version_number.push_back(version_number);
        intermediate_version_number.push_back(
            std::vector<Expr>(func.num_update_definitions(), Expr()));
    }
    assert(funcs.size() == intermediate_version_number.size());
    is_forward_canonicalizing_phase = false;

    // Bounds inference
    func_bounds = inference_bounds(output, output_bounds);

    // Create a stub for each function and each update to accumulate adjoints.
    for (int func_id = 0; func_id < (int)funcs.size(); func_id++) {
        const Func &func = funcs[func_id];
        for (int update_id = -1;
                update_id < func.num_update_definitions(); update_id++) {
            Func adjoint_func(
                func.name() + "_" + std::to_string(update_id + 1) + "_d_def__");
            bool is_final_output = func_id == (int)funcs.size() - 1 &&
                                   update_id == func.num_update_definitions() - 1;
            std::vector<Var> args = func.args();
            for (auto &arg : args) {
                if (arg.is_implicit()) {
                    // Replace implicit variables with non implicit ones
                    arg = Var();
                }
            }
            if (is_final_output) {
                adjoint_func(args) = adjoint(args);
            } else {
                // Initialize to 0
                if (func.values().size() == 1) {
                    adjoint_func(args) = make_const(func.values()[0].type(), 0.0);
                } else {
                    std::vector<Expr> init(func.values().size());
                    for (int i = 0; i < (int)init.size(); i++) {
                        init[i] = make_const(func.values()[i].type(), 0.0);
                    }
                    adjoint_func(args) = Tuple(init);
                }
            }
            FuncKey func_key{func.name(), update_id};
            assert(adjoint_funcs.find(func_key) == adjoint_funcs.end());
            adjoint_funcs[func_key] = adjoint_func;
        }
    }
    // Also create stubs for buffers referenced by the functions
    std::map<std::string, BufferInfo> called_buffers;
    for (int func_id = 0; func_id < (int)funcs.size(); func_id++) {
        const Func &func = funcs[func_id];
        std::map<std::string, BufferInfo> buffers = find_buffer_calls(func);
        called_buffers.insert(buffers.begin(), buffers.end());
    }
    for (const auto &it : called_buffers) {
        Func adjoint_func(it.first + "_d__");
        std::vector<Var> args;
        for (int i = 0; i < it.second.dimension; i++) {
            args.push_back(Var());
        }
        adjoint_func(args) = make_const(it.second.type, 0.0);
        FuncKey func_key{it.first, -1};
        if (adjoint_funcs.find(func_key) != adjoint_funcs.end()) {
            user_error << "Naming conflict between buffer and function:" <<
                it.first << "\n";
        }
        adjoint_funcs[func_key] = adjoint_func;
    }

    // Traverse functions from producers to consumers for reverse accumulation
    for (int func_id = funcs.size() - 1; func_id >= 0; func_id--) {
        const Func &func = funcs[func_id];
        current_func = func;

        FuncKey func_key{func.name(), func.num_update_definitions() - 1};
        // Set up boundary condition for the last adjoint
        Func &adjoint_func = adjoint_funcs[func_key];
        const Box &bounds = func_bounds[func.name()];

        // Save a pointer to the unbounded def. Useful for scheduling
        FuncKey unbounded_func_key{func.name() + "_unbounded", func_key.second};
        adjoint_funcs[unbounded_func_key] = adjoint_func;

        if (adjoint_func.values().size() == 1) {
            Type type = adjoint_func.values()[0].type();
            adjoint_func = BoundaryConditions::constant_exterior(
                adjoint_func, make_const(type, 0.0), box_to_vector(bounds),
                    adjoint_func.name() + "_ce");
        } else {
            std::vector<Expr> values(adjoint_func.values().size());
            for (int i = 0; i < (int)values.size(); i++) {
                values[i] = make_const(adjoint_func.values()[i].type(), 0.0);
            }
            adjoint_func = BoundaryConditions::constant_exterior(
                adjoint_func, Tuple(values), box_to_vector(bounds),
                adjoint_func.name() + "_ce");
        }

        // Traverse from the last update to first
        for (int update_id = func.num_update_definitions() - 1;
                update_id >= -1; update_id--) {
            current_update_id = update_id;
            FuncKey func_key{func.name(), update_id};
            internal_assert(func_bounds.find(func.name()) != func_bounds.end());
            current_has_intermediate_rdom = false;
            if (update_id >= 0) {
                internal_assert((int)intermediate_version_number[func_id].size() ==
                    func.num_update_definitions());
                current_has_intermediate_rdom =
                    intermediate_version_number[func_id][update_id].defined();
            }

            // Initialize the next adjoint function by 
            // propagating the adjoints to next update
            // Example:
            // f(x) = ...
            // f(1) = ... <- we're here
            // We have an adjoint for f(1) defined over the whole support of f
            // Now we want to initialize for the f(x) update
            // Need to propagate back to all x while masking 1
            // x -> next_args
            // 1 -> update_args
            auto mask_previous_update = [&] () {
                FuncKey prev_func_key{func.name(), update_id - 1};
                Func &prev_adjoint_func = adjoint_funcs[prev_func_key];
                std::vector<Var> prev_args = prev_adjoint_func.args();
                std::vector<Expr> update_args = func.update_args(update_id);
                // Replace implicit variables
                for (auto &arg : update_args) {
                    std::set<std::string> implicit_variables =
                        find_implicit_variables(arg);
                    for (const auto &var : implicit_variables) {
                        arg = substitute(var, prev_args[Var::implicit_index(var)], arg);
                    }
                }
                // Check if prev_args are the same as update_args
                // If they are the same simply set everything to zero
                bool is_noop = true;
                for (int i = 0 ; i < (int)prev_args.size(); i++) {
                    const Variable *update_var = update_args[i].as<Variable>();
                    if (update_var == nullptr || prev_args[i].name() != update_var->name) {
                        is_noop = false;
                    }
                }
                prev_adjoint_func = Func(prev_adjoint_func.name());
                if (!is_noop) {
                    // f'(x) = adjoint
                    prev_adjoint_func(prev_args) =
                        adjoint_funcs[func_key](prev_args);
                }
                if (func.values().size() == 1) {
                    Type type = func.values()[0].type();
                    prev_adjoint_func(update_args) = make_const(type, 0.0);
                } else {
                    std::vector<Expr> init(func.values().size());
                    for (int i = 0; i < (int)init.size(); i++) {
                        init[i] = make_const(func.values()[i].type(), 0.0);
                    }
                    prev_adjoint_func(update_args) = Tuple(init);
                }
            };        
            if (update_id >= 0 && !current_has_intermediate_rdom) {
                // Delay the masking if we're keeping track of intermediate values
                // Since in this case we are propagating to current update
                // instead of previous update.
                mask_previous_update();
            }

            // Now we want to propagate the derivatives at expression level
            // Topologically sort the expressions for each value in the tuple
            std::vector<Expr> expr_list;
            Tuple rhs_tuple =
                update_id < 0 ? func.values() : func.update_values(update_id);
            std::vector<const BaseExprNode *> output_exprs;
            const std::vector<Expr> &rhs_tuple_vector = rhs_tuple.as_vector();
            for (const auto &expr : rhs_tuple_vector) {
                std::vector<Expr> value_expr_list = sort_expressions(expr);
                expr_list.insert(
                    expr_list.end(), value_expr_list.begin(), value_expr_list.end());
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
            // f(x) = g(x)
            // d_g(x) = d_f(x) * df/dg
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

            // We propagate in two phases, the first phase only propagates
            // to self references, the second phase propagates to the rest
            { // First phase
                is_self_referencing_phase = true;
                expr_adjoints.clear();
                for (int i = 0; i < (int)output_exprs.size(); i++) {
                    expr_adjoints[output_exprs[i]] =
                        Call::make(adjoint_funcs[func_key].function(),
                            update_args, i);
                }

                // Traverse the expressions in reverse order
                for (auto it = expr_list.rbegin(); it != expr_list.rend(); it++) {
                    // Propagate adjoints
                    it->accept(this);
                }
            }
            if (current_has_intermediate_rdom) {
                // Now, if we use a RDom for keeping track
                // of intermediate values, the adjoints 
                // go for current update.
                // We want to propagate to previous adjoint
                // with the first element of the RDom.

                FuncKey prev_func_key{func_key.first, func_key.second - 1};
                // Recreate a new adjoint for previous update
                Func prev_adjoint;
                std::vector<Expr> args;
                args.reserve(adjoint_func.args().size());
                for (const auto &arg : adjoint_func.args()) {
                    args.push_back(arg);
                }
                std::vector<Expr> undefs;
                undefs.reserve(adjoint_func.outputs());
                std::vector<Expr> values = adjoint_func.values().as_vector();
                for (const auto &val : values) {
                    undefs.push_back(undef(val.type()));
                }
                prev_adjoint(args) = Tuple(undefs);
                args.back() =
                    simplify(intermediate_version_number[func_id][update_id] - 1);
                std::vector<Expr> calls;
                calls.reserve(values.size());
                for (int i = 0; i < (int)values.size(); i++) {
                    calls.push_back(Call::make(
                        adjoint_funcs[func_key].function(), args, i));
                }
                prev_adjoint(args) = Tuple(calls);
                adjoint_funcs[prev_func_key] = prev_adjoint;
                mask_previous_update();
            }
            { // Second phase
                is_self_referencing_phase = false;
                expr_adjoints.clear();
                for (int i = 0; i < (int)output_exprs.size(); i++) {
                    expr_adjoints[output_exprs[i]] =
                        Call::make(adjoint_funcs[func_key].function(),
                            update_args, i);
                }

                // Traverse the expressions in reverse order
                for (auto it = expr_list.rbegin(); it != expr_list.rend(); it++) {
                    // Propagate adjoints
                    it->accept(this);
                }
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

    // d/dx cast(x) = 1.f if op->type is float otherwise 0
    if (op->type.is_float()) {
        accumulate(op->value, make_const(op->type, 1.0));
    } else {
        accumulate(op->value, make_const(op->type, 0));
    }
}

void ReverseAccumulationVisitor::visit(const Variable *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // If the variable is a let variable, accumulates adjoints into the content
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
    accumulate(op->a,
        select(op->a <= op->b, adjoint, make_const(adjoint.type(), 0.0)));
    // d/db min(a, b) = b <= a ? 1 : 0
    accumulate(op->b,
        select(op->b <= op->a, adjoint, make_const(adjoint.type(), 0.0)));
}

void ReverseAccumulationVisitor::visit(const Max *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];

    // d/da max(a, b) = a >= b ? 1 : 0
    accumulate(op->a,
        select(op->a >= op->b, adjoint, make_const(adjoint.type(), 0.0)));
    // d/db max(a, b) = b >= a ? 1 : 0
    accumulate(op->b,
        select(op->b >= op->a, adjoint, make_const(adjoint.type(), 0.0)));
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
    accumulate(op->true_value,
        select(op->condition, adjoint, make_const(adjoint.type(), 0.0)));
    // d/dc select(a, b, c) = select(a, 0, 1)
    accumulate(op->false_value,
        select(op->condition, make_const(adjoint.type(), 0.0), adjoint));
}

void ReverseAccumulationVisitor::visit(const Call *op) {
    assert(expr_adjoints.find(op) != expr_adjoints.end());
    Expr adjoint = expr_adjoints[op];
    if (op->is_extern()) {
        // Math functions
        if (check_opname(op->name, "exp")) {
            // d/dx exp(x) = exp(x)
            accumulate(op->args[0], adjoint * exp(op->args[0]));
        } else if (check_opname(op->name, "log")) {
            // d/dx log(x) = 1 / x
            accumulate(op->args[0], adjoint / op->args[0]);
        } else if (check_opname(op->name, "sin")) {
            // d/dx sin(x) = cos(x)
            accumulate(op->args[0], adjoint * cos(op->args[0]));
        } else if (check_opname(op->name, "asin")) {
            // d/dx asin(x) = 1 / sqrt(1 - x^2)
            Expr one = make_const(op->type, 1.0);
            accumulate(op->args[0], adjoint / sqrt(one - op->args[0] * op->args[0]));
        } else if (check_opname(op->name, "cos")) {
            // d/dx cos(x) = -sin(x)
            accumulate(op->args[0], - adjoint * sin(op->args[0]));
        } else if (check_opname(op->name, "acos")) {
            // d/dx acos(x) = - 1 / sqrt(1 - x^2)
            Expr one = make_const(op->type, 1.0);
            accumulate(op->args[0], - adjoint / sqrt(one - op->args[0] * op->args[0]));
        } else if (check_opname(op->name, "tan")) {
            // d/dx tan(x) = 1 / cos(x)^2
            Expr c = cos(op->args[0]);
            accumulate(op->args[0], adjoint / (c * c));
        } else if (check_opname(op->name, "atan")) {
            // d/dx atan(x) = 1 / (1 + x^2)
            Expr one = make_const(op->type, 1.0);
            accumulate(op->args[0], adjoint / (one + op->args[0] * op->args[0]));
        } else if (check_opname(op->name, "atan2")) {
            Expr x2y2 = op->args[0] * op->args[0] + op->args[1] * op->args[1];
            // d/dy atan2(y, x) = x / (x^2 + y^2)
            accumulate(op->args[0], adjoint * op->args[1] / x2y2);
            // d/dx atan2(y, x) = -y / (x^2 + y^2)
            accumulate(op->args[1], -adjoint * op->args[0] / x2y2);
        } else if (check_opname(op->name, "sinh")) {
            // d/dx sinh(x) = cosh(x)
            accumulate(op->args[0], adjoint * cosh(op->args[0]));
        } else if (check_opname(op->name, "asinh")) {
            // d/dx asin(x) = 1 / sqrt(1 + x^2)
            Expr one = make_const(op->type, 1.0);
            accumulate(op->args[0], adjoint / sqrt(one + op->args[0] * op->args[0]));
        } else if (check_opname(op->name, "cosh")) {
            // d/dx cosh(x) = sinh(x)
            accumulate(op->args[0], adjoint * sinh(op->args[0]));
        } else if (check_opname(op->name, "acosh")) {
            // d/dx acosh(x) = 1 / (sqrt(x - 1) sqrt(x + 1)))
            Expr one = make_const(op->type, 1.0);
            accumulate(op->args[0],
                adjoint / (sqrt(op->args[0] - one) * sqrt(op->args[0] + one)));
        } else if (check_opname(op->name, "tanh")) {
            // d/dx tanh(x) = 1 / cosh(x)^2
            Expr c = cosh(op->args[0]);
            accumulate(op->args[0], adjoint / (c * c));
        } else if (check_opname(op->name, "atanh")) {
            // d/dx atanh(x) = 1 / (1 - x^2)
            Expr one = make_const(op->type, 1.0);
            accumulate(op->args[0], adjoint / (one - op->args[0] * op->args[0]));
        } else if (check_opname(op->name, "ceil")) {
            // TODO: d/dx = dirac(n) for n in Z ...
            accumulate(op->args[0], make_const(op->type, 0.0));
        } else if (check_opname(op->name, "floor")) {
            // TODO: d/dx = dirac(n) for n in Z ...
            accumulate(op->args[0], make_const(op->type, 0.0));
        } else if (check_opname(op->name, "round")) {
            accumulate(op->args[0], make_const(op->type, 0.0));
        } else if (check_opname(op->name, "trunc")) {
            accumulate(op->args[0], make_const(op->type, 0.0));
        } else if (check_opname(op->name, "sqrt")) {
            Expr half = make_const(op->type, 0.5);
            accumulate(op->args[0], adjoint * half / sqrt(op->args[0]));
        } else if (check_opname(op->name, "pow")) {
            Expr one = make_const(op->type, 1.0);
            accumulate(op->args[0],
                adjoint * op->args[1] * pow(op->args[0], op->args[1] - one));
            accumulate(op->args[1],
                adjoint * pow(op->args[0], op->args[1]) * log(op->args[0]));
        } else if (check_opname(op->name, "fast_inverse")) {
            // d/dx 1/x = -1/x^2
            Expr inv_x = fast_inverse(op->args[0]);
            accumulate(op->args[0], -adjoint * inv_x * inv_x);
        } else if (check_opname(op->name, "fast_inverse_sqrt")) {
            // d/dx x^(-0.5) = -0.5*x^(-1.5)
            Expr inv_sqrt_x = fast_inverse_sqrt(op->args[0]);
            Expr neg_half = make_const(op->type, -0.5);
            accumulate(op->args[0],
                neg_half * adjoint * inv_sqrt_x * inv_sqrt_x * inv_sqrt_x);
        } else if (op->name == "halide_print") {
            accumulate(op->args[0], make_const(op->type, 0.0));
        } else {
            internal_error << "The derivative of " << op->name <<
                " is not implemented.";
        }
    } else if (op->is_intrinsic()) {
        if (op->is_intrinsic(Call::abs)) {
            accumulate(op->args[0],
                adjoint*select(op->args[0] > 0,
                    make_const(op->type, 1.0), make_const(op->type, -1.0)));
        } else if (op->is_intrinsic(Call::lerp)) {
            // z = x * (1 - w) + y * w
            // dz/dx = 1 - w
            // dz/dy = w
            // dz/dw = y - x
            accumulate(op->args[0], adjoint * (make_const(op->type, 1.0) - op->args[2]));
            accumulate(op->args[1], adjoint * op->args[2]);
            accumulate(op->args[2], adjoint * (op->args[1] - op->args[0]));
        } else if (op->is_intrinsic(Call::likely)) {
            accumulate(op->args[0], adjoint);
        } else if (op->is_intrinsic(Call::return_second)) {
            accumulate(op->args[0], make_const(op->type, 0.0));
            accumulate(op->args[1], adjoint);
        } else if (op->is_intrinsic(Call::undef)) {
            // do nothing
        } else {
            user_warning << "Dropping gradients at call to " << op->name << "\n";
            for (const auto &arg : op->args) {
                accumulate(arg, make_const(op->type, 0.0));
            }
        }
    } else if (op->call_type == Call::Halide ||
               op->call_type == Call::Image) { // Halide function call or Halid buffer
        if (is_forward_canonicalizing_phase) {
            // Don't need to propagate through function in this phase, we're just
            // checking local derivatives
            return;
        }
        if (is_self_referencing_phase) {
            // We want to make sure we propagate to the self reference first.
            // In this phase only self reference is propagated
            if (!op->func.same_as(current_func.function().get_contents())) {
                return;
            }
        } else {
            // In the other phase we ignore the self reference
            if (op->func.same_as(current_func.function().get_contents())) {
                return;
            }
        }
        // Add Let expressions
        adjoint = add_let_expression(adjoint, let_var_mapping, let_variables);
        std::vector<Expr> lhs = op->args;
        for (int i = 0; i < (int)lhs.size(); i++) {
            lhs[i] = add_let_expression(lhs[i], let_var_mapping, let_variables);
        }
        Expr adjoint_before_canonicalize = adjoint;
        std::vector<Expr> lhs_before_canonicalize = lhs;

        // We create different functions for the initial condition and each update
        // When update i uses value from update i-1, we accumulate the
        // adjoints to update i-1
        // If target is the current function itself, send to previous update
        // e.g. f(x) = ...
        //      f(x) = f(x) + 1
        // For the one with non-commutative-associative reductions
        // e.g. f(x, ver) = ...
        //      f(x, 0) = ...
        //      f(x, r.x + 1) = f(x, r.x) * f(x, r.x) + g(r.x)
        // We propagate the whole r.x to the current update.
        // In addition, we propagate the first one (d_f(x, 0)) to the previous update,
        // by setting all reduction variables to their min() values.
        // Because only f(x, 0) comes from the last update, and
        // the rest belongs to the current update.
        // The above case will be handled by the caller, here we just
        // propagate to current update.
        // TODO: make the comments clearer and clean up the code
        FuncKey func_key;
        if (op->func.defined()) {
            Function func(op->func);
            func_key = func.name() != current_func.name() ?
                       FuncKey{func.name(), func.updates().size() - 1} :
                       FuncKey{func.name(), current_update_id - 1};
            if (current_has_intermediate_rdom && is_self_referencing_phase) {
                func_key = FuncKey{func.name(), current_update_id};
            }
        } else {
            func_key = FuncKey{op->name, -1};
        }
        assert(adjoint_funcs.find(func_key) != adjoint_funcs.end());
        Func& func_to_update = adjoint_funcs[func_key];
        assert(func_to_update.dimensions() == (int)lhs.size());

        bool debug_flag = false;

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
            std::set<std::string> implicit_variables =
                find_implicit_variables(adjoint);
            for (const auto &var : implicit_variables) {
                adjoint = substitute(
                    var, current_args[Var::implicit_index(var)], adjoint);
            }
        }

        // We want to do this:
        // func_to_update(op->args) += adjoint(current_update_args);
        // But op->args can be invalid lhs, need to canonicalize.
        // We canonicalize by first trying to substitute with pure variables.
        // If that fails we will replace variables on lhs with RDoms
        // (general scattering).

        // We try canonicalize the left hand side arguments (op->args)
        // so that it's always x, y, z, ...
        //
        // Given:
        // g(x, y, z) = f(x, y-1, z+1)
        // we get an invalid update:
        // f'(x, y - 1, z + 1) += g'(x, y, z)
        // Goal: rewrite to
        //  ==> f'(x, y, z) += g'(x, y+1, z-1)
        // (below we would call g and g' the "current function" and 
        //  we call f and d_f the "function to update")
        //
        // We do this by set up a new set of variables new_args
        // new_args contains a set of variable u0, u1, u2, ...
        // For each left hand side of the update (x, y - 1, z + 1 here),
        // we set up the equation u0 = x, u1 = y - 1, u2 = z + 1.
        // Then we solve for x, y, z and get x = u0, y = u1 + 1, z = u2 - 1
        // We would get f'(u0, u1, u2) += g'(u0, u1 + 1, u2 - 1)
        // We then substitute the original variable names back to get
        // f'(x, y, z) += g'(x, x + 1, z - 1)
        //
        // Currently we don't want to mess with system solving yet,
        // So we gather all arguments that contains multiple pure variables,
        // and invalidate all of them.
        // Inter-dependencies like:
        // g(x, y) = f(x * y, x + y)
        // can't be simplified.
        // In principle this can be inverted by solving a system of equations.
        // In this case we replace x and y with reduction variables that loop
        // through g's bounds
        // i.e.
        // f'(r.x * r.y, r.x + r.y) += g'(r.x, r.y)

        // Prepare a set of new substitution variables for func_to_update
        std::vector<Var> new_args;
        new_args.reserve(func_to_update.args().size());
        for (int arg_id = 0; arg_id < (int)func_to_update.args().size(); arg_id++) {
            new_args.push_back(Var("u" + std::to_string(arg_id) + "_"));
        }

        // Loop over the left hand side of the update, construct equations
        // and invert them.
        std::vector<bool> canonicalized(lhs.size(), false);
        std::set<std::string> canonicalized_vars;
        std::map<std::string, Var> lhs_substitute_map;
        for (int arg_id = 0; arg_id < (int)lhs.size(); arg_id++) {
            // Gather all pure variables at op->args[arg_id],
            // substitute them with new_args
            // For now only support single pure variable
            std::vector<std::string> variables =
                gather_variables(lhs[arg_id], vars_to_strings(current_args));
            if (variables.size() != 1) {
                continue;
            }

            bool solved;
            Expr result_rhs;
            std::tie(solved, result_rhs) =
                solve_inverse(new_args[arg_id] == lhs[arg_id],
                              new_args[arg_id].name(),
                              variables[0]);
            if (!solved) {
                continue;
            }

            // Replace pure variable with the reverse.
            // Make sure to also substitute predicates
            adjoint = substitute_rdom_predicate(variables[0], result_rhs, adjoint);

            // Since we successfully invert, the left hand side becomes
            // new_args
            lhs[arg_id] = new_args[arg_id];
            // Record that we sucessfully invert, for those we fail
            // we need to perform general scattering.
            canonicalized[arg_id] = true;
            canonicalized_vars.insert(variables[0]);
            lhs_substitute_map[variables[0]] = new_args[arg_id];
        }

        // Sometimes we have this kind of pathelogical case:
        // f(x, y) = ...
        // k(n) = f(g(n), n)
        // When we update d_f, we the second n would be replaced by y
        // We need to make sure we also update the call argument to g
        // adjoint is automatically handles in the loop above
        for (int i = 0; i < (int)lhs.size(); i++) {
            for (const auto &it : lhs_substitute_map) {
                lhs[i] = substitute(it.first, it.second, lhs[i]);
            }
        }

        // Sometimes the canonicalization above fails.
        // We replace the pure variables inside lhs with RDoms for general scattering
        std::vector<std::pair<Expr, Expr>> bounds;
        bounds.reserve(current_args.size());
        for (int arg_id = 0; arg_id < (int)current_args.size(); arg_id++) {
            bounds.push_back({
                current_bounds[arg_id].min,
                current_bounds[arg_id].max - current_bounds[arg_id].min + 1});
        }
        RDom r_bounds(bounds);
        for (int lhs_id = 0; lhs_id < (int)lhs.size(); lhs_id++) {
            if (!canonicalized[lhs_id]) {
                Expr lhs_arg = lhs[lhs_id];
                std::vector<std::string> variables =
                    gather_variables(lhs_arg, current_adjoint_func.function().args());
                RDom r(bounds);
                // For each variable found in lhs_arg, find the corresponding
                // bound (by looping through all variables) and substitute
                // with the bound reduction variable.
                for (int var_id = 0; var_id < (int)variables.size(); var_id++) {
                    for (int arg_id = 0; arg_id < (int)current_args.size(); arg_id++) {
                        if (current_args[arg_id].name() == variables[var_id] &&
                                canonicalized_vars.find(
                                    current_args[arg_id].name()) ==
                                    canonicalized_vars.end()) {
                            lhs[lhs_id] = substitute(variables[var_id],
                                                     r_bounds[arg_id],
                                                     lhs[lhs_id]);
                            adjoint = substitute(
                                variables[var_id], r_bounds[arg_id], adjoint);
                            break;
                        }
                    }
                }
            }
        }

        // For each free variable on the rhs, replace it with current bounds
        // e.g. we have in forward pass f(x, y) = g(x)
        //      then we would have g'(x) += f'(x, y) by now
        //      now we need to replace y with a reduction variable over f's bound
        //      x is automatically excluded since it's currently
        //      replaced by the new substitution variable e.g. u_0

        // First gather all free variables
        FuncBounds bounds_subset;
        std::vector<int> arg_id_to_substitute;
        bounds_subset.reserve(current_args.size());
        arg_id_to_substitute.reserve(current_args.size());
        for (int arg_id = 0; arg_id < (int)current_args.size(); arg_id++) {
            if (has_variable(adjoint, current_args[arg_id].name())) {
                const Interval &interval = current_bounds[arg_id];
                bounds_subset.emplace_back(
                    interval.min, interval.max - interval.min + 1);
                arg_id_to_substitute.push_back(arg_id);
            }
        }

        // Create a new RDom to loop over all free variables
        if (arg_id_to_substitute.size() > 0) {
            RDom r(bounds_subset);
            for (int i = 0; i < (int)arg_id_to_substitute.size(); i++) {
                int arg_id = arg_id_to_substitute[i];
                adjoint = substitute(current_args[arg_id].name(), r[i], adjoint);
            }
        }

        // General scattering simplification rules
        // For each expression in lhs, 
        // check if it is an expression of a single rvar and
        // spans the same interval of the function's bound
        // if so we can rewrite it back to pure variables
        // e.g.
        // f(r.x) = g(r.x)
        // => f(x) = g(x)
        //
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
            // f(r.x) = g(r.x)
            // => f(x) = g(x)
            if (var != nullptr && var->reduction_domain.defined() &&
                    var->reduction_domain.split_predicate().size() == 0) {
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
                // Check if the min/max of the rvariable equal to
                // the target function
                const Box &target_bounds = func_bounds[op->name];
                Interval t_interval = target_bounds[i];
                t_interval.min = simplify(t_interval.min);
                t_interval.max = simplify(t_interval.max);
                Interval r_interval(simplify(rvar.min),
                                    simplify(rvar.min + rvar.extent - 1));
                if (can_prove(r_interval.min <= t_interval.min &&
                              r_interval.max >= t_interval.max) && false) {
                    lhs[i] = func_to_update_args[i];
                    // Replace other occurence of rvar in lhs
                    for (int j = 0; j < (int)lhs.size(); j++) {
                        if (j != i) {
                            lhs[j] = simplify(substitute(
                                rvar.var, func_to_update_args[i], lhs[j]));
                        }
                    }
                    adjoint = simplify(substitute(
                        rvar.var, func_to_update_args[i], adjoint));
                }
            // f(4 * r.x + r.y) = g(r.x) + h(4 * r.x + r.y)
            // => f(x) = g(x/4) + h(x)
            } else if (add != nullptr &&
                       ((add->a.as<Mul>() != nullptr && 
                            add->b.as<Variable>() != nullptr) ||
                        (add->a.as<Variable>() != nullptr && 
                            add->b.as<Mul>() != nullptr))) {
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

        // We can only have one RDom for each update.
        // Therefore we have to merge RDoms on both lhs and rhs
        // To make use of better locality we preserve partial order
        std::map<std::string, ReductionVariableInfo> rvar_maps =
            gather_rvariables(adjoint);
        for (const auto &lhs_arg : lhs) {
            std::map<std::string, ReductionVariableInfo> maps =
                gather_rvariables(lhs_arg);
            rvar_maps.insert(maps.begin(), maps.end());
        }
        // Original set of reduction variables
        std::map<std::string, ReductionVariableInfo> org_rvar_maps =
            gather_rvariables(adjoint_before_canonicalize);
        for (const auto &lhs_arg : lhs_before_canonicalize) {
            std::map<std::string, ReductionVariableInfo> maps =
                gather_rvariables(lhs_arg);
            org_rvar_maps.insert(maps.begin(), maps.end());
        }
        // If the update is not commutative, we need to flip the 
        // original set of reduction variable
        if (current_has_intermediate_rdom) {
            // For each lhs
            for (auto &lhs_arg : lhs) {
                // For each original rvar
                for (const auto &it : org_rvar_maps) {
                    RVar r(it.second.domain, it.second.index);
                    Expr max = simplify(it.second.min + it.second.extent - 1);
                    // Replace the reduction with the flipped version
                    lhs_arg = substitute(it.first, max - r, lhs_arg);
                }
            }
            // For adjoint
            // For each original rvar
            for (const auto &it : org_rvar_maps) {
                RVar r(it.second.domain, it.second.index);
                Expr max = simplify(it.second.min + it.second.extent - 1);
                // Replace the reduction with the flipped version
                adjoint = substitute(it.first, max - r, adjoint);
            }
        }

        // Order: newly introduced rvar -> original rvar
        std::vector<ReductionVariableInfo> new_rvar_vec, old_rvar_vec;
        for (const auto &it : rvar_maps) {
            if (org_rvar_maps.find(it.first) == org_rvar_maps.end()) {
                new_rvar_vec.push_back(it.second);
            } else {
                old_rvar_vec.push_back(it.second);
            }
        }

        // Sort by index & domain
        auto cmp_rv = [] (const ReductionVariableInfo &rv0,
                          const ReductionVariableInfo &rv1) {
            ReductionDomain::Compare cmp;
            if (cmp(rv0.domain, rv1.domain)) {
                return true;
            } else {
                return rv0.index < rv1.index;
            }
        };
        std::sort(new_rvar_vec.begin(), new_rvar_vec.end(), cmp_rv);
        std::sort(old_rvar_vec.begin(), old_rvar_vec.end(), cmp_rv);
        // Flatten to an array
        std::vector<std::string> var_names;
        FuncBounds merged_bounds;
        for (const auto &it : new_rvar_vec) {
            var_names.push_back(it.name);
            merged_bounds.emplace_back(it.min, it.extent);
        }
        for (const auto &it : old_rvar_vec) {
            var_names.push_back(it.name);
            merged_bounds.emplace_back(it.min, it.extent);
        }
        // Produce final merged RDom
        RDom merged_r;
        if (merged_bounds.size() > 0) {
            merged_r = RDom(merged_bounds);
            // Transfer the predicate from old RDoms to merged RDom
            // Gather the set of RDoms
            std::set<ReductionDomain, ReductionDomain::Compare> rdoms;
            for (const auto &it : rvar_maps) {
                rdoms.insert(it.second.domain);
            }
            Expr rdom_predicate = Internal::UIntImm::make(UInt(1), 1);
            for (const auto &rdom : rdoms) {
                rdom_predicate = simplify(rdom_predicate && rdom.predicate());
            }
            // Reference to new RDom
            for (int rid = 0; rid < merged_r.dimensions(); rid++) {
                adjoint = substitute(var_names[rid], merged_r[rid], adjoint);
                for (auto &lhs_arg : lhs) {
                    lhs_arg = substitute(var_names[rid], merged_r[rid], lhs_arg);
                }
                rdom_predicate = substitute(
                    var_names[rid], merged_r[rid], rdom_predicate);
            }
            if (!is_const(rdom_predicate)) {
                for (int arg_id = 0; arg_id <
                        (int)func_to_update_args.size(); arg_id++) {
                    // Substitute new_args back to original variables
                    rdom_predicate = substitute(new_args[arg_id].name(),
                        func_to_update_args[arg_id], rdom_predicate);
                }
                merged_r.where(rdom_predicate);
            }
        }

        // Substitute new_args back to original variables
        for (int arg_id = 0; arg_id < (int)func_to_update_args.size(); arg_id++) {
            for (auto &lhs_arg : lhs) {
                lhs_arg = substitute(new_args[arg_id].name(),
                    func_to_update_args[arg_id], lhs_arg);
            }
            adjoint = substitute_rdom_predicate(
                new_args[arg_id].name(), func_to_update_args[arg_id], adjoint);
        }
        adjoint = simplify(adjoint);

        if (debug_flag) {
            debug(0) << "func_to_update.name():" << func_to_update.name() << "\n";
            debug(0) << "lhs after canonicalization:";
            for (const auto &arg : lhs) {
                debug(0) << " " << arg;
            }
            debug(0) << "\n";
            debug(0) << "adjoint after canonicalization:" << simplify(adjoint) << "\n";
        }

        // Finally we update the function definitions, 
        // possibly merge with previous updates
        auto can_merge = [&](Func &func_to_update,
                             const std::vector<Expr> &lhs) -> bool {
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
            // If previous update has different left hand side, don't merge
            for (int i = 0; i < (int)prev_lhs.size(); i++) {
                if (!equal(lhs[i], prev_lhs[i])) {
                    return false;
                }
            }
            // If previous update has a different set of reduction variables, 
            // don't merge
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

        // TODO: maybe do some analysis on lhs to avoid applying boundary conditions to
        //       function calls in adjoint
        if (!can_merge(func_to_update, lhs)) {
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
                    values[op->value_index] =
                        simplify(values[op->value_index] + adjoint);
                }
            }
        }
    } else {
        // TODO: let user provide derivatives for external functions
        internal_error << "Unknown call type of operation: " << op->name << "\n";
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
            return make_const(op->type, 0.0);
        }
    } else if (const Call *op = expr.as<Call>()) {
        if (op->is_extern()) {
            if (check_opname(op->name, "exp")) {
                // d/dx exp(f(x)) = exp(f(x)) f'
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return expr * d;
            } else if (check_opname(op->name, "log")) {
                // d/dx log(f(x)) = f' / f(x)
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return d / expr;
            } else if (check_opname(op->name, "sin")) {
                // d/dx sin(f(x)) = cos(f(x)) f'
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return cos(op->args[0]) * d;
            } else if (check_opname(op->name, "asin")) {
                // d/dx asin(f(x)) = f' / sqrt(1 - f(x)^2)
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr one = make_const(op->type, 1.0);
                return d / sqrt(one - op->args[0] * op->args[0]);
            } else if (check_opname(op->name, "cos")) {
                // d/dx cos(f(x)) = -sin(f(x)) f'
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return -sin(op->args[0]) * d;
            } else if (check_opname(op->name, "acos")) {
                // d/dx acos(f(x)) = -f' / sqrt(1 - f(x)^2)
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr one = make_const(op->type, 1.0);
                return -d / sqrt(one - op->args[0] * op->args[0]);
            } else if (check_opname(op->name, "tan")) {
                // d/dx tan(f(x)) = f' / cos^2(f(x))
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr cos_x = cos(op->args[0]);
                return d / (cos_x * cos_x);
            } else if (check_opname(op->name, "atan")) {
                // d/dx tan(f(x)) = f' / cos^2(f(x))
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr one = make_const(op->type, 1.0);
                return d / (op->args[0] * op->args[0] + one);
            } else if (check_opname(op->name, "atan2")) {
                // d/dx atan2(f(x), g(x)) =
                //   f' * (g(x) / (f(x)^2 + g(x)^2)) -
                //   g' * (f(x) / (f(x)^2 + g(x)^2))
                Expr d0 = forward_accumulation(op->args[0], tangents, scope);
                Expr d1 = forward_accumulation(op->args[1], tangents, scope);
                Expr norm = op->args[0] * op->args[0] + op->args[1] * op->args[1];
                return (d0 * op->args[1] - d1 * op->args[0]) / norm;
            } else if (check_opname(op->name, "sinh")) {
                // d/dx sinh(f(x)) = f'cosh(f(x))
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return d * cosh(op->args[0]);
            } else if (check_opname(op->name, "asinh")) {
                // d/dx asinh(f(x)) = f' / sqrt(f(x)^2 + 1)
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr one = make_const(op->type, 1.0);
                return d / sqrt(one + op->args[0] * op->args[0]);
            } else if (check_opname(op->name, "cosh")) {
                // d/dx cosh(f(x)) = f'sinh(f(x))
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return d * sinh(op->args[0]);
            } else if (check_opname(op->name, "acosh")) {
                // d/dx asinh(f(x)) = f' / sqrt((f(x) - 1) * (f(x) + 1))
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr one = make_const(op->type, 1.0);
                return d / sqrt((op->args[0] - one) * (op->args[0] + one));
            } else if (check_opname(op->name, "tanh")) {
                // d/dx sinh(f(x)) = f'/cosh(f(x))^2
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr cosh_x = cosh(op->args[0]);
                return d / (cosh_x * cosh_x);
            } else if (check_opname(op->name, "atanh")) {
                // d/dx sinh(f(x)) = f'/(1 - f(x)^2)
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr one = make_const(op->type, 1.0);
                return d / (one - op->args[0] * op->args[0]);
            } else if (check_opname(op->name, "ceil")) {
                return make_const(op->type, 0.0);
            } else if (check_opname(op->name, "floor")) {
                return make_const(op->type, 0.0);
            } else if (check_opname(op->name, "round")) {
                return make_const(op->type, 0.0);
            } else if (check_opname(op->name, "trunc")) {
                return make_const(op->type, 0.0);
            } else if (check_opname(op->name, "sqrt")) {
                // d/dx f(x)^(0.5) = 0.5 * f(x)^(-0.5) f'
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return (0.5f * d / expr);
            } else if (check_opname(op->name, "pow")) {
                // d/dx pow(f(x), g(x)) = pow(f(x), g(x)-1) *
                //                        (g(x) f'(x) + f(x) log(f(x))g'(x))
                Expr a = forward_accumulation(op->args[0], tangents, scope);
                Expr b = forward_accumulation(op->args[1], tangents, scope);
                return pow(op->args[0], op->args[1] - 1.f) *
                    (op->args[1] * a +
                     // Special hack: if g' == 0 then even if f == 0 the following term is 0
                     // basically we want -Inf * 0 = 0
                     select(b == 0.f,
                         make_const(op->type, 0.0),
                         op->args[0] * log(op->args[0]) * b));
            } else if (check_opname(op->name, "fast_inverse")) {
                // d/dx f(x)^(-1) = -f' * f(x)^(-2)
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr inv_x = fast_inverse(op->args[0]);
                return -d * inv_x * inv_x;
            } else if (check_opname(op->name, "fast_inverse_sqrt")) {
                // d/dx f(x)^(-0.5) = -0.5 * f' * f(x)^(-1.5)
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                Expr inv_sqrt_x = fast_inverse_sqrt(op->args[0]);
                Expr neg_half = make_const(op->type, -0.5);
                return neg_half * d * inv_sqrt_x * inv_sqrt_x * inv_sqrt_x;
            } else if (op->name == "halide_print") {
                return make_const(op->type, 0.0);
            } else {
                internal_error << "The derivative of " << op->name <<
                    " is not implemented.";
            }
        } else if (op->call_type == Call::Image || op->call_type == Call::Halide) {
            auto it = tangents.find(op->name);
            if (it != tangents.end()) {
                Func tangent = it->second;
                return tangent(op->args);
            } else {
                return make_const(op->type, 0.0);
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
                return -(op->args[2] - 1.f) * a + (op->args[1] - op->args[0]) * w + op->args[2] * b;
            } else if (op->is_intrinsic(Call::likely)) {
                Expr d = forward_accumulation(op->args[0], tangents, scope);
                return likely(d);
            } else if (op->is_intrinsic(Call::return_second)) {
                Expr d = forward_accumulation(op->args[1], tangents, scope);
                return d;
            } else if (op->is_intrinsic(Call::stringify)) {
                return make_const(op->type, 0.0);
            } else if (op->is_intrinsic(Call::undef)) {
                return make_const(op->type, 0.0);
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
        return make_const(expr.type(), 0.0);
    }
    return make_const(expr.type(), 0.0);
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
    adjoint(output.args()) = Internal::make_const(output.value().type(), 1.0);
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
                    Expr e = val;
                    for (const auto &it : options.variables) {
                        e = substitute(it.first, it.second, e);
                    }
                    Internal::debug(0) << " " << Internal::simplify(e);
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

template <typename T>
inline void CMP(int line_number, T x, T target, T threshold = T(1e-6)) {
    internal_assert(std::fabs((x) - (target)) < threshold) << \
        "Line " << line_number << ": Expected " <<
            (target) << " instead of " << (x) << "\n";    
}

inline void CMP(int line_number, float16_t x, float16_t target) {
    return CMP(line_number, x, target, float16_t(5e-3));
}

/**
 *  Check all dependencies of func, return true if any of the dependent func
 *  uses non pure variables on left hand side
 */
bool has_non_pure_update(const Func &func) {
    std::map<std::string, Function> env = find_transitive_calls(func.function());
    std::vector<std::string> order =
        realization_order({func.function()}, env).first;
    for (const auto &func_name : order) {
        Func func(env[func_name]);
        // For each update
        for (int id = 0; id < func.num_update_definitions(); id++) {
            // For each argument on left hand side
            const std::vector<Expr> &args = func.update_args(id);
            for (Expr arg : args) {
                if (arg.as<Variable>() == nullptr) {
                    return true;
                }
            }
        }
    }
    return false;
}

template <typename T>
void test_scalar() {
    { // Test + - * / const
        Func x("x");
        x() = Expr(T(5));
        Func y("y");
        y() = x() * x() - Expr(T(2)) * x() + Expr(T(5)) + Expr(T(3)) / x();
        Derivative d = propagate_adjoints(y);
        Func dx = d(x);
        Buffer<T> dydx = dx.realize();
        // y = x^2 - 2x + 5 + 3 / x
        // dydx = 2x - 2 - 3 / x^2 = 12 - 3 / 25
        CMP(__LINE__, dydx(0), T(8.0 - 3.0 / 25.0));
    }
    { // Test special functions
        Func x("x");
        x() = Expr(T(0.5));
        Func y("y");
        y() = sin(x()) +
              cos(x()) +
              tan(x()) +
              exp(x()) +
              log(x()) +
              sqrt(x()) +
              pow(x(), Expr(T(1.5))) +
              pow(Expr(T(1.5)), x()) +
              asin(x()) +
              Expr(T(1.2)) * acos(x()) +
              atan(x()) +
              atan2(x(), Expr(T(2))) +
              Expr(T(1.3)) * atan2(Expr(T(2)), x()) +
              sinh(x()) +
              Expr(T(1.2)) * cosh(x()) +
              tanh(x()) +
              asinh(x()) +
              acosh(x() + Expr(T(1))) +
              atanh(x());
        Derivative d = propagate_adjoints(y);
        Buffer<T> dydx = d(x).realize();
        // dydx = cos(x) -
        //        sin(x) +
        //        1 / cos(x)^2 +
        //        exp(x) +
        //        1/x +
        //        1 / (2 sqrt(x)) +
        //        1.5 * x^0.5 +
        //        (1.5^x) * log(1.5) +
        //        1 / sqrt(1 - x^2) -
        //        1.2 / sqrt(1 - x^2) +
        //        1 / (x^2 + 1) +
        //        2.f / (4.f + x^2) -
        //        x / (4.f + x^2) +
        //        cosh(x) +
        //        1.2 * sinh(x) +
        //        tanh(x) +
        //        1 / sqrt(x^2 + 1) +
        //        1 / (sqrt(x - 1) * sqrt(x + 1)) +
        //        1 / (1 - x^2)
        CMP(__LINE__, dydx(0),
            T(std::cos(0.5f) -
              std::sin(0.5f) +
              (1.f / (std::cos(0.5f) * std::cos(0.5f))) +
              std::exp(0.5f) +
              1.f / 0.5f +
              1.f / (2.f * std::sqrt(0.5f)) +
              1.5f * std::pow(0.5f, 0.5f) +
              std::log(1.5f) * std::pow(1.5f, 0.5f) +
              1.f / std::sqrt(1.f - 0.5f * 0.5f) -
              1.2f / std::sqrt(1.f - 0.5f * 0.5f) +
              (1.f / (0.5f * 0.5f + 1.f)) +
              2.f / (4.f + 0.5f * 0.5f) -
              1.3f * 2.f / (4.f + 0.5f * 0.5f) +
              std::cosh(0.5f) +
              1.2f * std::sinh(0.5f) +
              1.f / (std::cosh(0.5f) * std::cosh(0.5f)) +
              1.f / std::sqrt(0.5f * 0.5f + 1.f) +
              1.f / (std::sqrt(0.5f) * std::sqrt(2.5f)) +
              1.f / (1.f - 0.5f * 0.5f)));
    }
    { // Test fast inv
        Func x("x");
        x() = 2.5f;
        Func y("y");
        y() = fast_inverse(x()) + fast_inverse_sqrt(x());
        Derivative d = propagate_adjoints(y);
        Buffer<float> dydx = d(x).realize();
        // dy/dx = -1/x^2 - 1/(2*x^(3/2))
        CMP(__LINE__, dydx(0), -1.f/(2.5f * 2.5f) - 1.f/(2.f*std::pow(2.5f, 3.f/2.f)), 1e-3f);
    }
    { // Test floor ceil round trunc
        Func x("x");
        x() = Expr(T(2.5));
        Func y("y");
        y() = ceil(x()) + floor(x()) + round(x()) + trunc(x());
        Derivative d = propagate_adjoints(y);
        Buffer<T> dydx = d(x).realize();
        CMP(__LINE__, dydx(0), T(0));
    }
    { // Test max min
        Func x("x");
        x() = Expr(T(2.5));
        Func y("y");
        y() = Expr(T(2)) * max(x(), Expr(T(5))) +
              Expr(T(3)) * max(x(), Expr(T(1))) +
              Expr(T(5)) * min(x(), Expr(T(3))) +
              Expr(T(7)) * min(x(), Expr(T(2)));
        Derivative d = propagate_adjoints(y);
        Buffer<T> dydx = d(x).realize();
        CMP(__LINE__, dydx(0), T(8));
    }
    { // Test abs
        Func x("x");
        x() = Expr(T(-2.5));
        Func y("y");
        y() = Expr(T(2)) * abs(x()) + Expr(T(3)) * abs(-x());
        Derivative d = propagate_adjoints(y);
        Buffer<T> dydx = d(x).realize();
        // y = -2x - 3x = -5x, dy/dx = -5
        CMP(__LINE__, dydx(0), T(-5));
    }
    { // Test select
        Func x("x");
        x() = Expr(T(5));
        Func y("y");
        y() = select(x() > Expr(T(0)), Expr(T(2)) * x(), Expr(T(3)) * x()) +
              select(x() < Expr(T(0)), Expr(T(5)) * x(), Expr(T(7)) * x());
        Derivative d = propagate_adjoints(y);
        Buffer<T> dydx = d(x).realize();
        CMP(__LINE__, dydx(0), T(9));
    }
    { // Test lerp
        Func x("x");
        x() = Expr(T(2));
        Func y("y");
        y() = Expr(T(6));
        Func w("w");
        w() = Expr(T(0.1));
        Func z("z");
        // z = x * (1 - w) + y * w
        z() = lerp(x(), y(), w());
        Derivative d = propagate_adjoints(z);
        // dzdx = 1 - w
        Buffer<T> dzdx = d(x).realize();
        CMP(__LINE__, dzdx(0), T(0.9));
        // dzdy = w
        Buffer<T> dzdy = d(y).realize();
        CMP(__LINE__, dzdy(0), T(0.1));
        // dzdw = y - x
        Buffer<T> dzdw = d(w).realize();
        CMP(__LINE__, dzdw(0), T(4.0));
    }
}

void test_1d_box_no_clamp() {
    Var x("x");
    Buffer<float> input(3);
    input(0) = 1.f; input(1) = 2.f; input(2) = 3.f;
    Func blur("blur");
    blur(x) = input(x) + input(x + 1);
    RDom r(0, 2);
    Func f_loss("f_loss");
    f_loss() += blur(r.x) * blur(r.x);
    Derivative d = propagate_adjoints(f_loss);

    Buffer<float> blur_buf = blur.realize(2);
    // d loss / d blur = 2 * blur(x)
    Buffer<float> d_blur_buf = d(blur).realize(2);
    CMP(__LINE__, d_blur_buf(0), 2 * blur_buf(0));
    CMP(__LINE__, d_blur_buf(1), 2 * blur_buf(1));
    // d input(x) = d blur(x) + d blur(x - 1)
    Func d_input = d(input);
    // Every dependency of d_input should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_input)) <<
        "Function has non pure update\n";
    Buffer<float> d_input_buf = d_input.realize(3);
    CMP(__LINE__, d_input_buf(0), d_blur_buf(0));
    CMP(__LINE__, d_input_buf(1), d_blur_buf(0) + d_blur_buf(1));
    CMP(__LINE__, d_input_buf(2), d_blur_buf(1));
}

void test_1d_box() {
    Var x("x");
    Buffer<float> input(2);
    input(0) = 1.f; input(1) = 2.f;
    Func clamped("clamped");
    Expr clamped_x = Halide::clamp(x, 0, input.width() - 1);
    clamped(x) = input(clamped_x);
    Func blur("blur");
    blur(x) = clamped(x) + clamped(x + 1);
    RDom r(0, 2);
    Func f_loss("f_loss");
    f_loss() += blur(r.x) * blur(r.x);
    Derivative d = propagate_adjoints(f_loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;

    Buffer<float> blur_buf = blur.realize(2);
    // d loss / d blur = 2 * blur(x)
    Buffer<float> d_blur_buf = d(blur).realize(2);
    CMP(__LINE__, d_blur_buf(0), 2 * blur_buf(0));
    CMP(__LINE__, d_blur_buf(1), 2 * blur_buf(1));
    // d clamped(x) = d blur(x) + d blur(x - 1)
    Func d_clamped = d(clamped);
    // Every dependency of d_clamped should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_clamped)) <<
        "Function has non pure update\n";
    Buffer<float> d_clamped_buf = d_clamped.realize(3);
    CMP(__LINE__, d_clamped_buf(0), d_blur_buf(0));
    CMP(__LINE__, d_clamped_buf(1), d_blur_buf(0) + d_blur_buf(1));
    CMP(__LINE__, d_clamped_buf(2), d_blur_buf(1));
    // d input(clamp(x, 0, 1)) = d clamped (x)
    Buffer<float> d_input_buf = d(input).realize(2);
    CMP(__LINE__, d_input_buf(0), d_clamped_buf(0));
    CMP(__LINE__, d_input_buf(1), d_clamped_buf(1) + d_clamped_buf(2));
}

void test_2d_box() {
    Var x("x"), y("y");
    Buffer<float> input(5, 5, "input");
    for (int i = 0; i < input.width(); i++) {
        for (int j = 0; j < input.height(); j++) {
            input(i, j) = (i + 1) * (j + 2);
        }
    }
    Func clamped("clamped");
    Expr clamped_x = Halide::clamp(x, 0, input.width()-1);
    Expr clamped_y = Halide::clamp(y, 0, input.height()-1);
    clamped(x, y) = input(clamped_x, clamped_y);
    Func blur_x("blur_x");
    blur_x(x, y) = clamped(x, y) + clamped(x + 1, y) + clamped(x + 2, y);
    Func blur_y("blur_y");
    blur_y(x, y) = blur_x(x, y - 1) + blur_x(x, y) + blur_x(x, y + 1);

    RDom r(0, 5, 0, 5);
    Func loss("loss");
    loss() += blur_y(r.x, r.y) * blur_y(r.x, r.y);
    Derivative d = propagate_adjoints(loss);

    Buffer<float> blur_y_buf = blur_y.realize(5, 5);
    // d loss / d blur_y = 2 * blur_y(x, y)
    Buffer<float> d_blur_y_buf = d(blur_y).realize(5, 5);
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
    // d loss / d blur_x = d blur_y(x, y) + d blur_y(x, y - 1) + d blur_y(x, y + 1)
    Buffer<float> d_blur_x_buf = d(blur_x).realize(5, 5);
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            float target = d_blur_y_buf(x, y);
            if (y >= 1) {
                target += d_blur_y_buf(x, y - 1);
            }
            if (y < 4) {
                target += d_blur_y_buf(x, y + 1);
            }
            float diff = fabs(d_blur_x_buf(x, y) - target);
            internal_assert(diff < eps)
                << "Expected d_blur_x(" << x << ", " << y << ") to be " <<
                target << " instead of " << d_blur_x_buf(x, y) << "\n" ;
        }
    }
    Func d_clamped = d(clamped);
    // Every dependency of d_clamped should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_clamped)) <<
        "Function has non pure update\n";
    Buffer<float> d_clamped_buf = d_clamped.realize(5, 5);
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
            float diff = fabs(d_clamped_buf(x, y) - target);
            internal_assert(diff < eps)
                << "Expected d_clamped(" << x << ", " << y << ") to be " <<
                target << " instead of " << d_clamped_buf(x, y) << "\n" ;
        }
    }
}

void test_update() {
    Var x("x");
    Buffer<float> input(3);
    input(0) = 1.f; input(1) = 2.f; input(2) = 3.f;
    Func clamped("clamped");
    Expr clamped_x = Halide::clamp(x, 0, input.width() - 1);
    clamped(x) = input(clamped_x);
    Func blur("blur");
    blur(x) = clamped(x);
    blur(x) += clamped(x + 1);
    RDom r(0, 3);
    Func f_loss("f_loss");
    f_loss() += blur(r.x) * blur(r.x);
    Derivative d = propagate_adjoints(f_loss);

    Buffer<float> blur_buf = blur.realize(3);
    // d loss / d blur = 2 * blur(x)
    Buffer<float> d_blur_buf = d(blur).realize(3);

    CMP(__LINE__, d_blur_buf(0), 2 * blur_buf(0));
    CMP(__LINE__, d_blur_buf(1), 2 * blur_buf(1));
    CMP(__LINE__, d_blur_buf(2), 2 * blur_buf(2));
    Func d_clamped = d(clamped);
    // Every dependency of d_clamped should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_clamped)) <<
        "Function has non pure update\n";
    Buffer<float> d_clamped_buf = d_clamped.realize(3);
    CMP(__LINE__, d_clamped_buf(0), d_blur_buf(0));
    CMP(__LINE__, d_clamped_buf(1), d_blur_buf(0) + d_blur_buf(1));
    CMP(__LINE__, d_clamped_buf(2), d_blur_buf(1) + d_blur_buf(2));
}

void test_nonlinear_update() {
    Var x("x");
    Buffer<float> input(3);
    input(0) = 1.f; input(1) = 2.f; input(2) = 3.f;
    Func clamped("clamped");
    Expr clamped_x = Halide::clamp(x, 0, input.width() - 1);
    clamped(x) = input(clamped_x);
    Func update("update");
    update(x) = clamped(x);
    update(x) = update(x) * update(x) + clamped(x + 1);
    // update(x) = clamp(x)^2 + clamp(x + 1)
    RDom r(0, 3);
    Func loss("loss");
    loss() += update(r.x);
    Derivative d = propagate_adjoints(loss);
    Buffer<float> loss_buf = loss.realize();
    // loss = update(0) + update(1) + update(2)
    //      = 1^2 + 2 + 2^2 + 3 + 3^2 + 3 = 1 + 2 + 4 + 3 + 9 + 3 = 22
    CMP(__LINE__, loss_buf(0), 22.f);

    // d loss / d update = 1
    Buffer<float> d_update_buf = d(update).realize(3);
    CMP(__LINE__, d_update_buf(0), 1.f);
    CMP(__LINE__, d_update_buf(1), 1.f);
    CMP(__LINE__, d_update_buf(2), 1.f);
    Func d_clamped = d(clamped);
    // d_clamp(x) = 2 * clamp(x) * d_update(x) + d_update(x - 1)
    Buffer<float> d_clamped_buf = d_clamped.realize(3);
    CMP(__LINE__, d_clamped_buf(0), 2.f * input(0));
    CMP(__LINE__, d_clamped_buf(1), 2.f * input(1) + 1.f);
    CMP(__LINE__, d_clamped_buf(2), 2.f * input(2) + 1.f);
}

void test_rdom_conv() {
    Var x("x");
    Buffer<float> input(4);
    input(0) = 1.f; input(1) = 2.f; input(2) = 3.f; input(3) = 4.f;
    Func clamped("clamped");
    clamped(x) = input(Halide::clamp(x, 0, input.width() - 1));
    Buffer<float> kernel(2);
    kernel(0) = 2.f; kernel(1) = 1.f;
    Func convolved("convolved");
    RDom support(0, 2);
    convolved(x) += clamped(x + support) * kernel(support);
    RDom r(0, 4);
    Func f_loss("f_loss");
    f_loss() += convolved(r.x) * convolved(r.x);
    Derivative d = propagate_adjoints(f_loss);
    Buffer<float> convolved_buf = convolved.realize(4);
    // d loss / d blur = 2 * blur(x)
    Buffer<float> d_convolved_buf = d(convolved).realize(4);
    for (int i = 0; i < 4; i++) {
        CMP(__LINE__, d_convolved_buf(i), 2 * convolved_buf(i));
    }
    // d loss / d clamped = d_convolved convolve with flipped kernel
    Func d_clamped = d(clamped);
    // Every dependency of d_clamped should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_clamped)) <<
        "Function has non pure update\n";
    Buffer<float> d_clamped_buf = d_clamped.realize(4);
    for (int i = 0; i < 4; i++) {
        float target = d_convolved_buf(i) * kernel(0);
        if (i >= 1) {
            target += d_convolved_buf(i - 1) * kernel(1);
        }
        CMP(__LINE__, d_clamped_buf(i), target);
    }
    // loss = (k0 + 2k1)^2 + (2k0 + 3k1)^2 + (3k0 + 4k1)^2 + (4k0 + 4k1)^2
    //      = k0^2 + 4k0k1 + 4k1^2 + 4k0^2 + 12 k0k1 + 9k1^2 + 9k0^2 + 24 k0k1 + 16 k1^2 + 16k0^2 + 32k0k1 + 16k1^2
    //      = 30 k0^2 + 72 k0k1 + 45 k1^2
    // d loss / d kernel(0) = 2 * k0 * 30 + 72 * k1
    // d loss / d kernel(1) = 72 * k0 + 90 * k1
    Buffer<float> d_kernel = d(kernel).realize(2);
    CMP(__LINE__, d_kernel(0), 60.f * kernel(0) + 72.f * kernel(1));
    CMP(__LINE__, d_kernel(1), 72.f * kernel(0) + 90.f * kernel(1));
}

void test_nonlinear_order_dependent_rdom() {
    Var x("x");
    Buffer<float> in(2);
    for (int i = 0; i < 2; i++) {
        in(i) = (i + 2.f);
    }
    RDom r(in);
    Func f;
    f(x) = in(x);
    f(x) = f(x) * f(x) + in(r.x);

    Func loss("loss");
    loss() += f(r);
    Derivative d = propagate_adjoints(loss);

    // Manual backprop
    float f0 = in(0);
    float f1 = in(1);
    float f0_a = f0 * f0 + in(0);
    float f0_b = f0_a * f0_a + in(1);
    float f1_a = f1 * f1 + in(0);
    float f1_b = f1_a * f1_a + in(1);
    float loss_ = f0_b + f1_b;
    float df0_b = 1.f;
    float df1_b = 1.f;
    float df1_a = df1_b * 2.f * f1_a;
    float din1 = df1_b;
    float df1 = df1_a * 2.f * f1;
    float din0 = df1_a;
    float df0_a = df0_b * 2.f * f0_a;
    din1 += df0_b;
    float df0 = df0_a * 2.f * f0;
    din0 += df0_a;
    din1 += df1;
    din0 += df0;
    Buffer<float> loss_buf = loss.realize();
    CMP(__LINE__, loss_, loss_buf());
    Buffer<float> d_in = d(in).realize(2);
    CMP(__LINE__, d_in(0), din0);
    CMP(__LINE__, d_in(1), din1);
}

void test_order_dependent_rdom() {
    Var x("x");
    Buffer<float> coeffs(8);
    for (int i = 0; i < 8; i++) {
        coeffs(i) = (i + 1.f);
    }
    RDom r(coeffs);
    Func polynomial("polynomial");
    Expr fx = x / cast<float>(1023);
    // Horner's method
    polynomial(x) = 0.f;
    polynomial(x) = polynomial(x) * fx + coeffs(coeffs.dim(0).max() - r);

    RDom rd(0, 1024);
    Func loss("loss");
    loss() += polynomial(rd) / 1024.f;
    Derivative d = propagate_adjoints(loss);

    // poly(0) = coeff(n)
    // poly(1) = poly(0) * x + coeff(n-1)
    // poly(2) = poly(1) * x + coeff(n-2)
    // ...
    // poly(n) = poly(n-1) * x + coeff(0)
    // The adjoint is
    // d_coeffs(0) = \sum_x d_poly(x, n)
    // d_coeffs(1) = \sum_x d_poly(x, n-1)
    // ..
    // and
    // d_poly(x, n) = \sum_x 1
    // d_poly(x, n-1) = \sum_x x
    // d_poly(x, n-2) = \sum_x x^2
    // ...
    Buffer<float> d_coeffs = d(coeffs).realize(8);
    for (int i = 0; i < 8; i++) {
        float d = 0.f;
        for (int j = 0; j < 1024; j++) {
            d += std::pow(j / 1023.f, (float)i);
        }
        d /= 1024.f;
        CMP(__LINE__, d_coeffs(i), d);
    }
}

void test_1d_to_2d() {
    Var x("x"), y("y");
    Buffer<float> input(2);
    input(0) = 1.f; input(1) = 2.f;
    Func output("output");
    output(x, y) = (x + 1.f) * input(y);

    RDom r(0, 2, 0, 2);
    Func loss("loss");
    loss() += output(r.x, r.y) * output(r.x, r.y);
    Derivative d = propagate_adjoints(loss);

    // loss = 5i0^2 + 5i1^2
    // d loss / d i0 = 10i0 = 10
    // d loss / d i1 = 10i1 = 20
    Buffer<float> d_output = d(output).realize(2, 2);
    CMP(__LINE__, d_output(0, 0), 2.f);
    CMP(__LINE__, d_output(1, 0), 4.f);
    CMP(__LINE__, d_output(0, 1), 4.f);
    CMP(__LINE__, d_output(1, 1), 8.f);

    Func d_input = d(input);
    // Every dependency of d_input should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_input)) <<
        "Function has non pure update\n";
    Buffer<float> d_input_buf = d_input.realize(2);
    CMP(__LINE__, d_input_buf(0), 10.f);
    CMP(__LINE__, d_input_buf(1), 20.f);
}

void test_linear_resampling_1d() {
    // f(x) = i1(i0(x)) with linear resampling
    Var x("x");
    Buffer<float> input0(2);
    input0(0) = 0.3f; input0(1) = 1.8f;
    Buffer<float> input1(3);
    input1(0) = 1.0f; input1(1) = 2.0f; input1(2) = 4.0f;
    Func clamped0("clamped0");
    clamped0(x) = input0(Halide::clamp(x, 0, input0.width() - 1));
    Func clamped1("clamped1");
    clamped1(x) = input1(Halide::clamp(x, 0, input1.width() - 1));
    Expr gx = clamped0(x);
    Expr fx = cast<int>(clamp(floor(clamped0(x)), 0.f, 1.f));
    Expr cx = fx + 1;
    Expr wx = gx - fx;
    Func interpolate("interpolate");
    interpolate(x) = clamped1(fx) * (1.f - wx) + clamped1(cx) * wx;

    RDom r(0, 2);
    Func loss("loss");
    loss() += interpolate(r.x);
    Derivative d = propagate_adjoints(loss);

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

    Buffer<float> interpolate_buf = interpolate.realize(2);
    CMP(__LINE__, interpolate_buf(0), 1.3f);
    CMP(__LINE__, interpolate_buf(1), 3.6f);

    Buffer<float> d_clamped0 = d(clamped0).realize(2);
    CMP(__LINE__, d_clamped0(0), 1.f);
    CMP(__LINE__, d_clamped0(1), 2.f);

    Buffer<float> d_clamped1 = d(clamped1).realize(3);
    CMP(__LINE__, d_clamped1(0), 0.7f);
    CMP(__LINE__, d_clamped1(1), 0.5f);
    CMP(__LINE__, d_clamped1(2), 0.8f);
}

void test_linear_resampling_2d() {
    // f(x, y) = i1(i0(x), y) with linear resampling
    Var x("x"), y("y");
    Buffer<float> input0(2, 1);
    input0(0, 0) = 0.3f; input0(1, 0) = 1.8f;
    Buffer<float> input1(3, 1);
    input1(0, 0) = 1.0f; input1(1, 0) = 2.0f; input1(2, 0) = 4.0f;
    Func clamped0("clamped0");
    Expr clamped_x0 = Halide::clamp(x, 0, input0.width() - 1);
    Expr clamped_y0 = Halide::clamp(y, 0, input0.height() - 1);
    clamped0(x, y) = input0(clamped_x0, clamped_y0);
    Func clamped1("clamped1");
    Expr clamped_x1 = Halide::clamp(x, 0, input1.width() - 1);
    Expr clamped_y1 = Halide::clamp(y, 0, input1.height() - 1);
    clamped1(x, y) = input1(clamped_x1, clamped_y1);
    Expr gx = clamped0(x, y);
    Expr fx = cast<int>(clamp(floor(clamped0(x, y)), 0.f, 1.f));
    Expr cx = fx + 1;
    Expr wx = gx - fx;
    Func interpolate("interpolate");
    interpolate(x, y) = clamped1(fx, y) * (1.f - wx) + clamped1(cx, y) * wx;

    RDom r(0, 2, 0, 1);
    Func loss("loss");
    loss() += interpolate(r.x, r.y);
    Derivative d = propagate_adjoints(loss);

    // Same as test_linear_resampling_1d()
    Buffer<float> interpolate_buf = interpolate.realize(2, 1);
    CMP(__LINE__, interpolate_buf(0, 0), 1.3f);
    CMP(__LINE__, interpolate_buf(1, 0), 3.6f);

    Buffer<float> d_clamped0 = d(clamped0).realize(2, 1);
    CMP(__LINE__, d_clamped0(0, 0), 1.f);
    CMP(__LINE__, d_clamped0(1, 0), 2.f);

    Buffer<float> d_clamped1 = d(clamped1).realize(3, 1);
    CMP(__LINE__, d_clamped1(0, 0), 0.7f);
    CMP(__LINE__, d_clamped1(1, 0), 0.5f);
    CMP(__LINE__, d_clamped1(2, 0), 0.8f);
}

void test_sparse_update() {
    Var x("x");
    Buffer<float> input(3);
    input(0) = 1.f; input(1) = 2.f; input(2) = 3.f;
    Func f_input("f_input");
    f_input(x) = input(x);
    Func output("output");
    output(x) = f_input(x);
    output(1) = 0.f;
    // XXX: if we write input(1) Halide returns a float
    // which means it is impossible to propagate to input,
    // so we need a surrogate f_input such that f_input(1) is symbolic
    output(2) = 2.f * f_input(1);

    Func loss("loss");
    RDom r(0, 3);
    loss() += output(r.x);
    Derivative d = propagate_adjoints(loss);

    Buffer<float> d_input = d(input).realize(3);
    CMP(__LINE__, d_input(0), 1.0f);
    CMP(__LINE__, d_input(1), 2.0f);
    CMP(__LINE__, d_input(2), 0.0f);
}

void test_histogram() {
    Var x("x");
    Buffer<int> input(4, "input");
    input(0) = 2; input(1) = 2; input(2) = 1; input(3) = 3;
    Buffer<float> k(5, "k");
    k(0) = 0.5f; k(1) = 1.f; k(2) = 1.5f; k(3) = 2.f; k(4) = 2.5f;
    Func output("output");
    output(x) = 0.f;
    RDom r(input);
    output(clamp(input(r), 0, 3)) += k(r);

    Func loss("loss");
    RDom rd(input);
    loss() += output(rd.x) * cast<float>(rd.x + 1);
    Derivative d = propagate_adjoints(loss);

    // d_output(2) -> d_k(0)
    // d_output(2) -> d_k(1)
    // d_output(1) -> d_k(2)
    // d_output(3) -> d_k(3)
    Buffer<float> d_k = d(k).realize(5);
    CMP(__LINE__, d_k(0), 3.0f);
    CMP(__LINE__, d_k(1), 3.0f);
    CMP(__LINE__, d_k(2), 2.0f);
    CMP(__LINE__, d_k(3), 4.0f);
    CMP(__LINE__, d_k(4), 0.0f);
}

void test_rdom_update() {
    Var x("x");
    Buffer<float> input(3);
    input(0) = 1.f; input(1) = 2.f; input(2) = 3.f;
    Func output("output");
    RDom r0(1, 2), r1(3, 4);
    output(x) = input(x);
    output(r0) = input(r0 - 1);
    output(r1) = 0.f;

    Func loss("f_loss");
    RDom r_target(0, 5);
    loss() += output(r_target);
    Derivative d = propagate_adjoints(loss);

    Buffer<float> d_input = d(input).realize(3);
    CMP(__LINE__, d_input(0), 2.0f);
    CMP(__LINE__, d_input(1), 1.0f);
    CMP(__LINE__, d_input(2), 0.0f);
}

void test_repeat_edge() {
    Var x("x");
    Buffer<float> input(2);
    input(0) = 1.f; input(1) = 2.f;
    Func clamped = BoundaryConditions::repeat_edge(input);
    Func blur("blur");
    blur(x) = clamped(x) + clamped(x + 1);
    RDom r(0, 3);
    Func loss("loss");
    loss() += blur(r.x);
    Derivative d = propagate_adjoints(loss);
    // loss = (i0 + i1) + (i1 + i1) + (i1 + i1) = i0 + 5 * i1

    Buffer<float> d_blur_buf = blur.realize(3);
    Buffer<float> d_input_buf = d(input).realize(2);
    // d loss / d i0 = 1
    // d loss / d i1 = 5
    CMP(__LINE__, d_input_buf(0), 1.f);
    CMP(__LINE__, d_input_buf(1), 5.f);
}

void test_constant_exterior() {
    Var x("x");
    Buffer<float> input(2);
    input(0) = 1.f; input(1) = 2.f;
    Func clamped = BoundaryConditions::constant_exterior(input, 0.f);
    Func blur("blur");
    blur(x) = clamped(x) + clamped(x + 1);
    RDom r(0, 3);
    Func loss("loss");
    loss() += blur(r.x);
    Derivative d = propagate_adjoints(loss);
    // loss = (i0 + i1) + i1 = i0 + 2 * i1

    Buffer<float> d_blur_buf = blur.realize(3);
    Buffer<float> d_input_buf = d(input).realize(2);
    // d loss / d i0 = 1
    // d loss / d i1 = 2
    CMP(__LINE__, d_input_buf(0), 1.f);
    CMP(__LINE__, d_input_buf(1), 2.f);
}

void test_repeat_image() {
    Var x("x");
    Buffer<float> input(2);
    input(0) = 1.f; input(1) = 2.f;
    Func clamped = BoundaryConditions::repeat_image(input);
    Func blur("blur");
    blur(x) = clamped(x) + clamped(x + 1);
    RDom r(0, 3);
    Func loss("loss");
    loss() += blur(r.x);
    Derivative d = propagate_adjoints(loss);
    // loss = (i0 + i1) + (i1 + i0) + (i0 + i1) = 3 * i0 + 3 * i1

    Buffer<float> d_blur_buf = blur.realize(3);
    Buffer<float> d_input_buf = d(input).realize(2);
    // d loss / d i0 = 3
    // d loss / d i1 = 3
    CMP(__LINE__, d_input_buf(0), 3.f);
    CMP(__LINE__, d_input_buf(1), 3.f);
}

void test_mirror_image() {
    Var x("x");
    Buffer<float> input(2);
    input(0) = 1.f; input(1) = 2.f;
    Func clamped = BoundaryConditions::mirror_image(input);
    Func blur("blur");
    blur(x) = clamped(x) + clamped(x + 1);
    RDom r(0, 3);
    Func loss("loss");
    loss() += blur(r.x);
    Derivative d = propagate_adjoints(loss);
    // loss = (i0 + i1) + (i1 + i1) + (i1 + i0) = 2 * i0 + 4 * i1

    Buffer<float> d_blur_buf = blur.realize(3);
    Buffer<float> d_input_buf = d(input).realize(2);
    // d loss / d i0 = 2
    // d loss / d i1 = 4
    CMP(__LINE__, d_input_buf(0), 2.f);
    CMP(__LINE__, d_input_buf(1), 4.f);
}

void test_mirror_interior() {
    Var x("x");
    Buffer<float> input(2);
    input(0) = 1.f; input(1) = 2.f;
    Func clamped = BoundaryConditions::mirror_interior(input);
    Func blur("blur");
    blur(x) = clamped(x) + clamped(x + 1);
    RDom r(0, 3);
    Func loss("loss");
    loss() += blur(r.x);
    Derivative d = propagate_adjoints(loss);
    // loss = (i0 + i1) + (i1 + i0) + (i0 + i1) = 3 * i0 + 3 * i1

    Buffer<float> d_blur_buf = blur.realize(3);
    Buffer<float> d_input_buf = d(input).realize(2);
    // d loss / d i0 = 3
    // d loss / d i1 = 3
    CMP(__LINE__, d_input_buf(0), 3.f);
    CMP(__LINE__, d_input_buf(1), 3.f);
}

void test_second_order() {
    Var x("x");
    Func input("input");
    input() = 1.f;
    Func polynomial("polynomial");
    // x^2 + 3x + 4.f
    polynomial() = input() * input() + 3.f * input() + 4.f;
    Derivative d = propagate_adjoints(polynomial);
    Func d_input = d(input);
    Derivative d2 = propagate_adjoints(d_input);
    Func d2_input = d2(input);

    Buffer<float> buf = d_input.realize();
    Buffer<float> buf2 = d2_input.realize();
    // d/dx = 2x + 3
    CMP(__LINE__, buf(0), 5.f);

    // d^2/dx^2 = 2
    CMP(__LINE__, buf2(0), 2.f);
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
    conv(x) += input_re(x + rc - 1) * kernel(rc);
    RDom rl(0, 9);
    Func loss0("loss0");
    loss0() += pow(conv(rl) - target(rl), 2.f);
    Derivative d = propagate_adjoints(loss0);
    Func d_input = d(input);
    Func loss1("loss1");
    loss1() += d_input(rl);
    Derivative d2 = propagate_adjoints(loss1);

    Buffer<float> conv_buf = conv.realize(9);
    Buffer<float> d_conv_buf = d(conv).realize(9);
    // d_conv(x) = 2 * (conv(x) - target(x))
    for (int i = 0; i < 9; i++) {
        CMP(__LINE__, d_conv_buf(i), 2.f * (conv_buf(i) - target(i)));
    }
    // d_input(x) = d_conv(x + 1) + d_conv(x) + d_conv(x - 1)
    Buffer<float> d_input_buf = d_input.realize(10);
    CMP(__LINE__, d_input_buf(0), d_conv_buf(0) + d_conv_buf(1));
    for (int i = 1; i < 8; i++) {
        CMP(__LINE__, d_input_buf(i), d_conv_buf(i+1) + d_conv_buf(i) + d_conv_buf(i-1));
    }
    CMP(__LINE__, d_input_buf(8), d_conv_buf(7) + d_conv_buf(8));
    CMP(__LINE__, d_input_buf(9), d_conv_buf(8));
    Buffer<float> d2_conv_buf = d2(conv).realize(9);
    // d2_conv(x) = 6
    for (int i = 0; i < 8; i++) {
        CMP(__LINE__, d2_conv_buf(i), 6.f);
    }
    CMP(__LINE__, d2_conv_buf(8), 4.f);
    // d2_input(x) = d2_conv(x + 1) + d2_conv(x) + d2_conv(x - 1)
    Buffer<float> d2_input_buf = d2(input).realize(10);
    CMP(__LINE__, d2_input_buf(0), 2.f * d2_conv_buf(0) + d2_conv_buf(1));
    for (int i = 1; i <= 7; i++) {
        CMP(__LINE__, d2_input_buf(i), d2_conv_buf(i) + d2_conv_buf(i + 1) + d2_conv_buf(i - 1));
    }
    CMP(__LINE__, d2_input_buf(8), d2_conv_buf(8) + d2_conv_buf(7));
    CMP(__LINE__, d2_input_buf(9), d2_conv_buf(8));
}

void test_implicit_vars() {
    Var x("x");
    Buffer<float> input(2);
    input(0) = 1.f; input(1) = 2.f;
    Func copy("copy");
    copy(_) = input(_);
    RDom r(0, 2);
    Func loss("loss");
    loss() += copy(r.x);
    Derivative d = propagate_adjoints(loss);

    Func d_input = d(input);
    // Every dependency of d_input should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_input)) <<
        "Function has non pure update\n";
    Buffer<float> d_input_buf = d_input.realize(2);
    CMP(__LINE__, d_input_buf(0), 1.f);
    CMP(__LINE__, d_input_buf(1), 1.f);
    Func d_copy = d(copy);
    // Every dependency of d_copy should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_copy)) <<
        "Function has non pure update\n";
    Buffer<float> d_copy_buf = d_copy.realize(2);
    CMP(__LINE__, d_copy_buf(0), 1.f);
    CMP(__LINE__, d_copy_buf(1), 1.f);
}

void test_tuple() {
    Var x("x");
    Buffer<float> input(3);
    input(0) = 1.f; input(1) = 2.f; input(2) = 3.f;
    Func tuple("tuple");
    tuple(x) = Tuple(input(x), input(x + 1));
    tuple(x) += Tuple(1.f, 1.f);
    Func reduce("reduce");
    reduce(x) = tuple(x)[0] + tuple(x)[1];
    RDom r(0, 2);
    Func loss("loss");
    loss() += reduce(r.x);
    Derivative d = propagate_adjoints(loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;
    // tuple(0) = {1, 2}
    // tuple(1) = {2, 3}
    // reduce(0) = 3
    // reduce(1) = 5
    // loss = reduce(0) + reduce(1)
    //      = tuple(0)[0] + tuple(0)[1] + tuple(1)[0] + tuple(1)[1]
    //      = input(0) + input(1) * 2 + input(2)

    Func d_tuple = d(tuple);
    // Every dependency of d_tuple should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_tuple)) <<
        "Function has non pure update\n";
    Realization d_tuple_buf = d_tuple.realize(2);
    Buffer<float> d_tuple_buf_0 = d_tuple_buf[0];
    Buffer<float> d_tuple_buf_1 = d_tuple_buf[1];
    CMP(__LINE__, d_tuple_buf_0(0), 1.f);
    CMP(__LINE__, d_tuple_buf_0(1), 1.f);
    CMP(__LINE__, d_tuple_buf_1(0), 1.f);
    CMP(__LINE__, d_tuple_buf_1(1), 1.f);

    Func d_input = d(input);
    // Every dependency of d_input should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_input)) <<
        "Function has non pure update\n";
    Buffer<float> d_input_buf = d_input.realize(3);
    CMP(__LINE__, d_input_buf(0), 1.f);
    CMP(__LINE__, d_input_buf(1), 2.f);
    CMP(__LINE__, d_input_buf(2), 1.f);
}

void test_floor_ceil() {
    Var x("x");
    Buffer<float> input(3);
    input(0) = 1.f; input(1) = 2.f; input(2) = 3.f;
    Func floor_output("floor_output");
    floor_output(x) = input(cast<int>(floor(x / 4.f)));
    Func ceil_output("ceil_output");
    ceil_output(x) = input(cast<int>(ceil(x / 4.f)));
    Func output("output");
    output(x) = ceil_output(x) + floor_output(x);
    RDom r(0, 8);
    Func loss("loss");
    loss() += output(r.x);
    Derivative d = propagate_adjoints(loss);
    // floor_output(0~3) == input[0]
    // floor_output(4~7) == input[1]
    // ceil_output(0) == input[0]
    // ceil_output(1~4) == input[1]
    // ceil_output(5~7) = input[2]
    Buffer<float> d_input_buf = d(input).realize(3);

    CMP(__LINE__, d_input_buf(0), 5.f);
    CMP(__LINE__, d_input_buf(1), 8.f);
    CMP(__LINE__, d_input_buf(2), 3.f);
}

void test_downsampling() {
    Var x("x");
    Buffer<float> input(10);
    for (int i = 0; i < 10; i++) {
        input(i) = float(i);
    }
    Func output("output");
    RDom r(0, 4);
    output(x) += input(4 * x + r);
    RDom r_loss(0, 2);
    Func loss("loss");
    loss() += output(r_loss);
    Derivative d = propagate_adjoints(loss);
    // output(0) = \sum input(0~4)
    // output(1) = \sum input(5~8)
    Func d_input = d(input);
    // Every dependency of d_tuple should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_input)) <<
        "Function has non pure update\n";
    Buffer<float> d_input_buf = d_input.realize(10);

    for (int i = 0; i < 8; i++) {
        CMP(__LINE__, d_input_buf(i), 1.f);
    }
    CMP(__LINE__, d_input_buf(8), 0.f);
    CMP(__LINE__, d_input_buf(9), 0.f);
}

void test_upsampling() {
    Var x("x");
    Buffer<float> input(4);
    for (int i = 0; i < 4; i++) {
        input(i) = float(i);
    }
    Func output("output");
    output(x) = input(x / 4);
    RDom r_loss(0, 16);
    Func loss("loss");
    loss() += output(r_loss);
    Derivative d = propagate_adjoints(loss);
    Func d_input = d(input);
    // Every dependency of d_tuple should only use pure variables in lhs
    internal_assert(!has_non_pure_update(d_input)) <<
        "Function has non pure update\n";
    Buffer<float> d_input_buf = d_input.realize(4);

    for (int i = 0; i < 4; i++) {
        CMP(__LINE__, d_input_buf(i), 4.f);
    }
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
    loss() += pow(output(r.x, r.y) - target(r.x, r.y), 2);
    Derivative d = propagate_adjoints(loss);
    Func d_input = d(input);
    Buffer<float> d_input_buf = d_input.realize(5, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            CMP(__LINE__, d_input_buf(i, j), 2.f * (input(i, j) - target(j, i)));
        }
    }
}

void test_change_var() {
    Var x("x"), y("y"), a("a"), b("b");
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
    Func xy_func("xy");
    xy_func(x, y) = input(x, y);
    Func ab_func("ab");
    ab_func(a, b) = xy_func(a, b);
    RDom r(0, 5, 0, 5);
    Func loss("loss");
    loss() += pow(ab_func(r.x, r.y) - target(r.x, r.y), 2);
    Derivative d = propagate_adjoints(loss);
    Func d_input = d(input);
    Buffer<float> d_input_buf = d_input.realize(5, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            CMP(__LINE__, d_input_buf(i, j), 2.f * (input(i, j) - target(j, i)));
        }
    }
}

void test_rdom_predicate() {
    Var x("x"), y("y");
    Buffer<float> input(7, 7);
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            input(i, j) = float(i + j);
        }
    }
    RDom r(0, 7, 0, 7);
    r.where((r.x - 3)*(r.x - 3) + (r.y - 3)*(r.y - 3) <= 10);
    Func circle;
    circle(x, y) = input(x, y);
    circle(r.x, r.y) *= 2.f;

    RDom r_full(0, 7, 0, 7);
    Func loss("loss");
    loss() += circle(r_full.x, r_full.y);
    Derivative d = propagate_adjoints(loss);
    Func d_input = d(input);
    Buffer<float> d_input_buf = d_input.realize(7, 7);
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            bool in_circle =
                (i - 3)*(i - 3) + (j - 3)*(j - 3) <= 10;
            if (in_circle) {
                CMP(__LINE__, d_input_buf(i, j), 2.f);
            } else {
                CMP(__LINE__, d_input_buf(i, j), 1.f);
            }
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
    output(x) += input(x + r);
    Func d_input("d_input");
    d_input(x) = 1.f;
    Func d_output = propagate_tangents(output, {{input.name(), d_input}});
    // d_output(x) = \sum d_input(x + r)
    Buffer<float> d_output_buf = d_output.realize(5);

    for (int i = 0; i < 5; i++) {
        CMP(__LINE__, d_output_buf(i), 2.f);
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
        CMP(__LINE__, d_output_buf(i), 2.f * (output_buf(i) - target(i)));
    }
    Func d_input = d(input);
    Buffer<float> d_input_buf = d_input.realize(10);
    // d_input(x) = d_output(x) + d_output(x - 1)
    CMP(__LINE__, d_input_buf(0), d_output_buf(0));
    for (int i = 1; i < 9; i++) {
        CMP(__LINE__, d_input_buf(i), d_output_buf(i) + d_output_buf(i - 1));
    }
    CMP(__LINE__, d_input_buf(9), d_output_buf(8));
    Func d2_output = propagate_tangents(d_output, {{input.name(), d_input}});
    Buffer<float> d2_output_buf = d2_output.realize(9);
    // d2_output(x) = 2 * (d_input(x) + d_input(x + 1))
    for (int i = 0; i < 9; i++) {
        CMP(__LINE__, d2_output_buf(i), 2.f * (d_input_buf(i) + d_input_buf(i + 1)));
    }
    Func d2_input = propagate_tangents(d_input, {{input.name(), d_input}});
    Buffer<float> d2_input_buf = d2_input.realize(10);
    // d2_input(x) = d2_output(x) + d2_output(x - 1)
    CMP(__LINE__, d2_input_buf(0), d2_output_buf(0));
    for (int i = 1; i < 9; i++) {
        CMP(__LINE__, d2_input_buf(i), d2_output_buf(i) + d2_output_buf(i - 1));
    }
    CMP(__LINE__, d2_input_buf(9), d2_output_buf(8));
}

void derivative_test() {
    test_simple_bounds_inference();
    test_simple_bounds_inference_update();
    test_scalar<float>();
    test_scalar<double>();
    test_1d_box_no_clamp();
    test_1d_box();
    test_2d_box();
    test_update();
    test_nonlinear_update();
    test_rdom_conv();
    test_order_dependent_rdom();
    test_nonlinear_order_dependent_rdom();
    test_1d_to_2d();
    test_linear_resampling_1d();
    test_linear_resampling_2d();
    test_sparse_update();
    test_histogram();
    test_rdom_update();
    test_repeat_edge();
    test_constant_exterior();
    test_repeat_image();
    test_mirror_image();
    test_mirror_interior();
    test_second_order();
    test_second_order_conv();
    test_implicit_vars();
    test_tuple();
    test_floor_ceil();
    test_downsampling();
    test_upsampling();
    test_transpose();
    test_change_var();
    test_rdom_predicate();
    test_forward();
    test_reverse_forward();
    debug(0) << "Derivative test passed\n";
}

} // namespace Internal
} // namespace Halide
