#include "Derivative.h"

#include "Simplify.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IREquality.h"
#include "Error.h"
#include "runtime/printer.h"
#include <iostream>

namespace Halide {
namespace Internal {

class VariableFinder : public IRGraphVisitor {
public:
    bool find(const Expr &expr, const Var &var) {
        visited.clear();
        var_name = var.name();
        found = false;
        expr.accept(this);
        return found;
    }

    void visit(const Variable *op) {
        if (op->name == var_name) {
            found = true;
        }
    }

private:
    std::string var_name;
    bool found;
};


class VariableReplacer : public IRMutator {
public:
     Expr replace(const Expr &expr, const std::string &replaced_var_name_, const Expr &replace_expr_) {
        replaced_var_name = replaced_var_name_;
        replace_expr = replace_expr_;
        return mutate(expr);
    }

    void visit(const Variable *op) {
        if (op->name == replaced_var_name) {
            expr = replace_expr;
        } else {
            expr = op;
        }
    }

private:
    std::string replaced_var_name;
    Expr replace_expr;
};


Expr inverse(const Var &var, const Expr &expr) {
    // TODO: replace with a full visitor
    VariableFinder finder;
    if (expr.get()->node_type == IRNodeType::Add) {
        const Add *op = expr.as<Add>();
        bool in_a = finder.find(op->a, var);
        bool in_b = finder.find(op->b, var);
        if (in_a && !in_b) {
            return inverse(var, op->a) - op->b;
        } else if (in_b && !in_a) {
            return inverse(var, op->b) - op->a;
        }
    } else if (expr.get()->node_type == IRNodeType::Sub) {
        const Sub *op = expr.as<Sub>();
        bool in_a = finder.find(op->a, var);
        bool in_b = finder.find(op->b, var);
        if (in_a && !in_b) {
            return inverse(var, op->a) + op->b;
        } else if (in_b && !in_a) {
            return inverse(var, op->b) - op->a;
        }
    } else if (expr.get()->node_type == IRNodeType::Max) {
        const Max *op = expr.as<Max>();
        bool in_a = finder.find(op->a, var);
        bool in_b = finder.find(op->b, var);
        if (in_a && !in_b) {
            return max(inverse(var, op->a), op->b);
        } else if (in_b && !in_a) {
            return max(op->a, inverse(var, op->b));
        }
    } else if (expr.get()->node_type == IRNodeType::Min) {
        const Min *op = expr.as<Min>();
        bool in_a = finder.find(op->a, var);
        bool in_b = finder.find(op->b, var);
        if (in_a && !in_b) {
            return min(inverse(var, op->a), op->b);
        } else if (in_b && !in_a) {
            return min(op->a, inverse(var, op->b));
        }
    } else if (expr.get()->node_type == IRNodeType::Variable) {
        return expr;
    }
    assert(false);
    return expr;
}

std::pair<Expr, Expr> get_min_max_bounds(const Expr &expr, const std::vector<Var> &current_args,
                                         const RDom &current_bounds, const int index) {
    if (expr.get()->node_type == IRNodeType::Add) {
        const Add *op = expr.as<Add>();
        const std::pair<Expr, Expr> a_bounds = get_min_max_bounds(op->a, current_args, current_bounds, index);
        const std::pair<Expr, Expr> b_bounds = get_min_max_bounds(op->b, current_args, current_bounds, index);
        debug(0) << "  " << index << " bounds for Add\n";
        return {a_bounds.first + b_bounds.first, a_bounds.second + b_bounds.second};
    } else if (expr.get()->node_type == IRNodeType::Sub) {
        const Sub *op = expr.as<Sub>();
        const std::pair<Expr, Expr> a_bounds = get_min_max_bounds(op->a, current_args, current_bounds, index);
        const std::pair<Expr, Expr> b_bounds = get_min_max_bounds(op->b, current_args, current_bounds, index);
        debug(0) << "  " << index << " bounds for Sub\n";
        return {a_bounds.first - b_bounds.second, a_bounds.second - b_bounds.first};
    } else if (expr.get()->node_type == IRNodeType::Variable) {
        const Variable *var = expr.as<Variable>();
        if (var->reduction_domain.defined()) {
            ReductionVariable rvar = var->reduction_domain.domain()[index];
            debug(0) << "  " << index << " bounds for Rvar\n";
            return {rvar.min, rvar.min + rvar.extent - 1};
        } else {
            debug(0) << "  " << index << " bounds for Var\n";
            for (int i = 0; i < (int)current_args.size(); i++) {
                if (current_args[i].name() == var->name) {
                    return {current_bounds[i].min(), current_bounds[i].extent()};
                }
            }
        }
    } else if (expr.get()->node_type == IRNodeType::Max) {
        const Max *op = expr.as<Max>();
        const std::pair<Expr, Expr> a_bounds = get_min_max_bounds(op->a, current_args, current_bounds, index);
        const std::pair<Expr, Expr> b_bounds = get_min_max_bounds(op->b, current_args, current_bounds, index);
        debug(0) << "  " << index << " bounds for Max\n";
        return {max(a_bounds.first, b_bounds.first), max(a_bounds.second, b_bounds.second)};
    } else if (expr.get()->node_type == IRNodeType::Min) {
        const Min *op = expr.as<Min>();
        const std::pair<Expr, Expr> a_bounds = get_min_max_bounds(op->a, current_args, current_bounds, index);
        const std::pair<Expr, Expr> b_bounds = get_min_max_bounds(op->b, current_args, current_bounds, index);
        debug(0) << "  " << index << " bounds for Min\n";
        return {min(a_bounds.first, b_bounds.first), min(a_bounds.second, b_bounds.second)};
    } else if (expr.get()->node_type == IRNodeType::IntImm) {
        debug(0) << "  " << index << " bounds for IntImm\n";
        return {expr, expr};
    }

    internal_error << "Can't infer bounds, Expr type not handled\n";
    return std::pair<Expr, Expr>();
}

std::pair<Expr, Expr> merge_bounds(const std::pair<Expr, Expr> &bounds0, const std::pair<Expr, Expr> &bounds1) {
    return {simplify(min(bounds0.first, bounds1.first)), simplify(max(bounds0.second, bounds1.second))};
};

/** An IR graph visitor that gather the function DAG and sort them in reverse topological order
 */
class FunctionSorter : public IRGraphVisitor {
public:
    void sort(const Expr &expr);
    void sort(const Func &func);

    std::vector<Func> get_functions() const {
        return functions;
    }

    void visit(const Call *op);

private:
    std::vector<Func> functions;
    std::set<std::string> traversed_functions;
};

void FunctionSorter::sort(const Expr &expr) {
    visited.clear();
    expr.accept(this);
}

void FunctionSorter::sort(const Func &func) {
    traversed_functions.insert(func.name());
    functions.push_back(Func(func));
    // Traverse from the last update to first
    for (int update_id = func.num_update_definitions() - 1; update_id >= -1; update_id--) {
        if (update_id >= 0) {
            func.update_value(update_id).accept(this);
        } else {
            func.value().accept(this);
        }
    }
}

void FunctionSorter::visit(const Call *op) {
    if (op->call_type == Call::Halide) {
        Func func(Function(op->func));
        if (traversed_functions.find(func.name()) != traversed_functions.end()) {
            return;
        }
        sort(func);
        return;
    }

    for (size_t i = 0; i < op->args.size(); i++) {
        include(op->args[i]);
    }
}


/** An IR graph visitor that gather the expression DAG and sort them in topological order
 */
class ExpressionSorter : public IRGraphVisitor {
public:
    void sort(const Expr &expr);

    std::vector<Expr> get_expr_list() const {
        return expr_list;
    }

    void visit(const Call *op);
protected:
    void include(const Expr &e);
private:
    std::vector<Expr> expr_list;
};

void ExpressionSorter::sort(const Expr &e) {
    visited.clear();
    expr_list.clear();
    e.accept(this);
    expr_list.push_back(e);
}

void ExpressionSorter::visit(const Call *op) {
    // No point visiting the arguments of a Halide func or an image
    if (op->call_type == Call::Halide || op->call_type == Call::Image) {
        return;
    }

    for (size_t i = 0; i < op->args.size(); i++) {
        include(op->args[i]);
    }
}


void ExpressionSorter::include(const Expr &e) {
    if (visited.count(e.get())) {
        return;
    } else {
        visited.insert(e.get());
        e.accept(this);
        expr_list.push_back(e);
        return;
    }
}

typedef std::vector<std::pair<Expr, Expr>> FuncBounds;

/**
 *  Visit function calls and determine their bounds.
 *  So when we do f(x, y) = ... we know what the loop bounds are
 */
class BoundsInferencer : public IRVisitor {
public:
    void inference(const Expr &expr);
    void inference(const Func &func);

    void visit(const Call *op);

    std::map<std::string, RDom> get_func_bounds() const {
        // TODO(mgharbi): don't recompute that all the time..
        std::map<std::string, RDom> ret;
        // Convert to an Rdom
        for(auto b: func_bounds) { 
          debug(0) << "Computed bounds for " << b.first << ":\n";
          FuncBounds min_extent_bounds;
          min_extent_bounds.reserve(b.second.size());
          for (int i = 0; i < (int)b.second.size(); ++i) {
            Expr lower_bound = simplify(b.second[i].first);
            Expr extent = simplify(b.second[i].second - lower_bound+1);
            min_extent_bounds.push_back(std::make_pair(lower_bound, extent));
            debug(0) << "  arg" << i << " ("  << lower_bound << ", " << extent << ")\n";
          }
          ret[b.first] = RDom(min_extent_bounds);
        }
        return ret;
    }

private:
    int recursion_depth = 0;
    std::map<std::string, FuncBounds> func_bounds;
    std::set<std::string> traversed_functions;
    std::vector<Var> current_args;
    RDom current_bounds;
};

void BoundsInferencer::inference(const Expr &expr) {
    // visited.clear();
    expr.accept(this);
}

void BoundsInferencer::inference(const Func &func) {
    RDom previous_bounds = current_bounds;
    std::vector<Var> previous_args = current_args;
    current_bounds = RDom(func_bounds[func.name()]);
    current_args = func.args();
    
    // Traverse from the last update to first
    traversed_functions.insert(func.name());
    for (int update_id = func.num_update_definitions() - 1; update_id >= -1; update_id--) {
        if (update_id >= 0) {
            func.update_value(update_id).accept(this);
        } else {
            func.value().accept(this);
        }
    }
    current_args = previous_args;
    current_bounds = previous_bounds;
}

void BoundsInferencer::visit(const Call *op) {
    if (op->call_type == Call::Halide) {
        Func func(Function(op->func));
        debug(0) << recursion_depth << " Visiting " << func.name() << "\n";

        FuncBounds arg_bounds;
        arg_bounds.reserve(op->args.size());
        for (int i = 0; i < (int)op->args.size(); i++) {
            std::pair<Expr, Expr> min_max_bounds = get_min_max_bounds(op->args[i], current_args, current_bounds, i);
            arg_bounds.push_back(min_max_bounds);
        }

        // Update function bounds
        if (func_bounds.find(func.name()) != func_bounds.end()) {
            FuncBounds prev_bounds = func_bounds[func.name()];
            assert(arg_bounds.size() == prev_bounds.size());
            for (int i = 0; i < (int)arg_bounds.size(); i++) {
                arg_bounds[i] = merge_bounds(prev_bounds[i], arg_bounds[i]);
            }
            debug(0) << "  Updated function bounds:\n";
        }
        for (int i = 0; i < (int)arg_bounds.size(); i++) {
          debug(0) << "    arg" << i << " (" 
                   << arg_bounds[i].first << ", "
                   << arg_bounds[i].second << ")\n";
        }

        func_bounds[func.name()] = arg_bounds;

        // if (traversed_functions.find(func.name()) != traversed_functions.end()) {
        //     // already traversed
        //     debug(0) << "  already traversed.\n";
        //     return;
        // }

        recursion_depth += 1;
        inference(func);
        recursion_depth -= 1;
        return;
    }

    for (size_t i = 0; i < op->args.size(); i++) {
        // include(op->args[i]);
        op->args[i].accept(this);
    }
}


/** An IR visitor that computes the derivatives through reverse accumulation
 */
class ReverseAccumulationVisitor : public IRVisitor {
public:
    void propagate_adjoints(const Expr &output, const std::vector<Func> &funcs);
    std::map<std::string, Func> get_adjoint_funcs() const { return adjoint_funcs; }

protected:
    void visit(const Cast *op);
    void visit(const Variable *op);
    void visit(const Add *op);
    void visit(const Sub *op);
    void visit(const Mul *op);
    void visit(const Div *op);
    void visit(const Min *op);
    void visit(const Max *op);
    void visit(const Call *op);
    void visit(const Let *op);

private:
    void accumulate(const Expr &stub, const Expr &adjoint);

    std::map<const BaseExprNode *, Expr> accumulated_adjoints;
    std::map<std::string, Func> adjoint_funcs;
    Func tmp_adjoint_func;
    std::map<std::string, Expr> let_var_mapping;
    std::map<std::string, RDom> func_bounds;
    RDom current_bounds;
    std::vector<Var> current_args;
    std::string current_func_name;
};


void ReverseAccumulationVisitor::propagate_adjoints(const Expr &output, const std::vector<Func> &funcs) {
    if (funcs.size() == 0) {
        debug(0) << "ReverseAccumulationVisitor: no functions to backpropagate to.\n";
        return;
    }

    BoundsInferencer bounds_inferencer;
    debug(0) << "ReverseAccumulationVisitor: infering bounds.\n";
    bounds_inferencer.inference(output);
    func_bounds = bounds_inferencer.get_func_bounds();

    // Create a stub for each function to accumulate adjoints
    for (int i = 0; i < (int)funcs.size(); i++) {
        Func adjoint_func(funcs[i].name() + "_d__");
        adjoint_func(funcs[i].args()) = 0.f;
        adjoint_funcs[funcs[i].name()] = adjoint_func;
    }

    // Propagate output
    ExpressionSorter sorter;
    sorter.sort(output);
    std::vector<Expr> expr_list = sorter.get_expr_list();
    accumulate(output, 1.f);

    // Traverse the expressions in reverse order
    for (auto it = expr_list.rbegin(); it != expr_list.rend(); it++) {
        // Propagate adjoints
        it->accept(this);
    }

    // Traverse functions
    for (int i = 0; i < (int)funcs.size(); i++) {
        const Func &func = funcs[i];
        current_func_name = func.name();

        // Traverse from the last update to first
        for (int update_id = func.num_update_definitions() - 1; update_id >= -1; update_id--) {
            // Topologically sort the expressions
            ExpressionSorter sorter;
            if (update_id >= 0) {
                sorter.sort(func.update_value(update_id));
            } else {
                sorter.sort(func.value());
            }

            // TODO: take lhs other than (x, y, z) into account
            assert(func_bounds.find(func.name()) != func_bounds.end());
            current_bounds = func_bounds[func.name()];
            current_args = func.args();

            std::vector<Expr> expr_list = sorter.get_expr_list();
            // Retrieve previously propagated adjoint
            if (update_id == func.num_update_definitions() - 1) {
                std::vector<Expr> args;
                for (const auto &arg : func.args()) {
                    args.push_back(arg);
                }
                accumulated_adjoints[(const BaseExprNode *)expr_list.back().get()] =
                    Call::make(adjoint_funcs[func.name()].function(), args);
            }

            // Propagate to this temporary Func if we use the same function during update
            tmp_adjoint_func = Func(func.name() + "_d__");
            tmp_adjoint_func(func.args()) = 0.f;

            // Traverse the expressions in reverse order
            for (auto it = expr_list.rbegin(); it != expr_list.rend(); it++) {
                // Propagate adjoints
                it->accept(this);
            }

            // Add back the Func
            Func &adjoint_func = adjoint_funcs[func.name()];
            tmp_adjoint_func(adjoint_func.args()) += adjoint_func(adjoint_func.args());
            adjoint_funcs[func.name()] = tmp_adjoint_func;
        }
    }
}

void ReverseAccumulationVisitor::accumulate(const Expr &stub, const Expr &adjoint) {
    const BaseExprNode *stub_ptr = (const BaseExprNode *)stub.get();
    if (accumulated_adjoints.find(stub_ptr) == accumulated_adjoints.end()) {
        accumulated_adjoints[stub_ptr] = adjoint;
    } else {
        accumulated_adjoints[stub_ptr] += adjoint;
    }
}

void ReverseAccumulationVisitor::visit(const Cast *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    // d/dx cast(x) = 1
    accumulate(op->value, adjoint);
}

void ReverseAccumulationVisitor::visit(const Variable *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    auto it = let_var_mapping.find(op->name);
    if (it != let_var_mapping.end()) {
        accumulate(it->second, Let::make(op->name, it->second, adjoint));
    }
}

void ReverseAccumulationVisitor::visit(const Add *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    // d/da a + b = 1
    accumulate(op->a, adjoint);
    // d/db a + b = 1
    accumulate(op->b, adjoint);
}

void ReverseAccumulationVisitor::visit(const Sub *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    // d/da a - b = 1
    accumulate(op->a, adjoint);
    // d/db a - b = -1
    accumulate(op->b, -adjoint);
}

void ReverseAccumulationVisitor::visit(const Mul *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    // d/da a * b = b
    accumulate(op->a, adjoint * op->b);
    // d/db a * b = a
    accumulate(op->b, adjoint * op->a);
}

void ReverseAccumulationVisitor::visit(const Div *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    // d/da a / b = 1 / b
    accumulate(op->a, adjoint / op->b);
    // d/db a / b = - a / b^2
    accumulate(op->b, - adjoint * op->a / (op->b * op->b));
}

void ReverseAccumulationVisitor::visit(const Min *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    // d/da min(a, b) = a <= b ? 1 : 0
    accumulate(op->a, select(op->a <= op->b, adjoint, 0.f));
    // d/db min(a, b) = b <= a ? 1 : 0
    accumulate(op->b, select(op->b <= op->a, adjoint, 0.f));
}

void ReverseAccumulationVisitor::visit(const Max *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    // d/da max(a, b) = a >= b ? 1 : 0
    accumulate(op->a, select(op->a >= op->b, adjoint, 0.f));
    // d/db max(a, b) = b >= a ? 1 : 0
    accumulate(op->b, select(op->b >= op->a, adjoint, 0.f));
}

void ReverseAccumulationVisitor::visit(const Call *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];
    if (op->name == "exp_f32") {
        // d/dx exp(x) = exp(x)
        for (size_t i = 0; i < op->args.size(); i++) {
            accumulate(op->args[i], adjoint * exp(op->args[i]));
        }
    }

    if (op->func.defined()) {
        // This is a Halide function call
        Function func(op->func);
        // Gather the domain variables of the function
        std::vector<std::string> func_args = func.args();
        std::vector<Var> args;
        std::for_each(func_args.begin(), func_args.end(),
                      [&args](const std::string &name){ args.push_back(Var(name)); });
        // We are scattering to this function
        debug(0) << "Scattering to " << func.name() << "\n";
        debug(0) << "op->args:" << "\n";
        for (const auto &arg : op->args) {
            debug(0) << arg << "\n";
        }
        debug(0) << "adjoint is:" << adjoint << "\n";
        Func& func_to_update = func.name() != current_func_name ?
            adjoint_funcs[func.name()] : tmp_adjoint_func;
        // We want to do this:
        // func_to_update(op->args) += adjoint;
        // But op->args can be invalid lhs, need to canonicalize

        VariableFinder finder;
        VariableReplacer replacer;
        assert(func_bounds.find(func.name()) != func_bounds.end());

        // We canonicalize the left hand side arguments (op->args) so that it's always x, y, z, ...
        for (int i = 0; i < (int)op->args.size(); i++) {
            if (!finder.find(op->args[i], args[i])) {
                // When an argument x doesn't appear in op->args,
                // all x in adjoint needs to be replaced by a RDom looping through the bounds
                // of the current function
                if (finder.find(adjoint, args[i])) {
                    adjoint = replacer.replace(adjoint, args[i].name(), current_bounds[i]);
                }
                // If it's a RVar, we need to replace it with the non-reduction argument
                if (op->args[i].get()->node_type == IRNodeType::Variable) {
                    const Variable *var = op->args[i].as<Variable>();
                    if (var->reduction_domain.defined()) {
                        adjoint = replacer.replace(adjoint, var->name, args[i]);
                    }
                }
            } else {
                // Apply the inverse to rhs
                Expr inverse_arg = inverse(args[i], op->args[i]);
                adjoint = replacer.replace(adjoint, args[i].name(), inverse_arg);
            }
        }

        debug(0) << "adjoint after canonicalization:" << adjoint << "\n";
        func_to_update(args) += adjoint;
        debug(0) << "print(func_to_update)" << "\n";
        print_func(func_to_update);
    }
}

void ReverseAccumulationVisitor::visit(const Let *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    accumulate(op->body, adjoint);
    let_var_mapping[op->name] = op->value;
}

} // namespace Internal


std::map<std::string, Func> propagate_adjoints(const Expr &output) {
    Internal::FunctionSorter sorter;
    Internal::debug(0) << "Propagate: Sorting functions" << "\n";
    sorter.sort(output);
    std::vector<Func> funcs = sorter.get_functions();
    Internal::debug(0) << "Propagate: Sorted Func list:" << "\n";
    for (const auto &func : funcs) {
        Internal::debug(0) << "  . " << func.name() << "\n";
    }
    Internal::ReverseAccumulationVisitor visitor;
    visitor.propagate_adjoints(output, funcs);
    return visitor.get_adjoint_funcs();
}

void print_func(const Func &func) {
    Internal::debug(0) << "Printing function:" << func.name() << "\n";
    Internal::FunctionSorter sorter;
    sorter.sort(func);
    std::vector<Func> funcs = sorter.get_functions();
    for (int i = (int)funcs.size() - 1; i >= 0; i--) {
        Func &func = funcs[i];
        Internal::debug(0) << "  funcs[" << i << "]: " << func.name() << "\n";
        for (int update_id = -1; update_id < func.num_update_definitions(); update_id++) {
            if (update_id >= 0) {
                Internal::debug(0) << "    update:" << func.update_value(update_id) << "\n";
            } else {
                Internal::debug(0) << "    init:" << func.value() << "\n";
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
    Expr loss = blur_y(r.x, r.y);

    BoundsInferencer bounds_inferencer;
    bounds_inferencer.inference(loss);
    std::map<std::string, RDom> bounds = bounds_inferencer.get_func_bounds();

    internal_assert(equal(bounds["blur_y"][0].min(), 0)) 
      << "Expected 0 instead of " << bounds["blur_y"][0].min() << "\n" ;
    internal_assert(equal(bounds["blur_y"][0].extent(), width-2)) 
      << "Expected " << width-2  << " instead of " << bounds["blur_y"][0].extent() << "\n" ;
    internal_assert(equal(bounds["blur_y"][1].min(), 0)) 
      << "Expected 0 instead of " << bounds["blur_y"][1].min() << "\n" ;
    internal_assert(equal(bounds["blur_y"][1].extent(), height-2)) 
      << "Expected " << height-2  << " instead of " << bounds["blur_y"][1].extent() << "\n" ;

    internal_assert(equal(bounds["blur_x"][0].min(), 0)) 
      << "Expected 0 instead of " << bounds["blur_x"][0].min() << "\n" ;
    internal_assert(equal(bounds["blur_x"][0].extent(), width-2)) 
      << "Expected " << width-2  << " instead of " << bounds["blur_x"][0].extent() << "\n" ;
    internal_assert(equal(bounds["blur_x"][1].min(), 0)) 
      << "Expected 0 instead of " << bounds["blur_x"][1].min() << "\n" ;
    internal_assert(equal(bounds["blur_x"][1].extent(), height)) 
      << "Expected " << height  << " instead of " << bounds["blur_x"][1].extent() << "\n" ;

    internal_assert(equal(bounds["input"][0].min(), 0)) 
      << "Expected 0 instead of " << bounds["input"][0].min() << "\n" ;
    internal_assert(equal(bounds["input"][0].extent(), width)) 
      << "Expected " << width  << " instead of " << bounds["input"][0].extent() << "\n" ;
    internal_assert(equal(bounds["input"][1].min(), 0)) 
      << "Expected 0 instead of " << bounds["input"][1].min() << "\n" ;
    internal_assert(equal(bounds["input"][1].extent(), height)) 
      << "Expected " << height  << " instead of " << bounds["input"][1].extent() << "\n" ;
}

void derivative_test() {
  test_simple_bounds_inference();
  debug(0) << "Derivative test passed\n";
}
} // namespace Internal


} // namespace Halide
