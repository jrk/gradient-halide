#include "Derivative.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Error.h"
#include "runtime/printer.h"
#include <iostream>

namespace Halide {
namespace Internal {

class VariableFinder : public IRGraphVisitor {
public:
    bool find(const Expr &expr, const Var &var) {
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
        replced_var_name = replaced_var_name_;
        replace_expr = replace_expr_;
        return mutate(expr);
    }

    void visit(const Variable *op) {
        if (op->name == replced_var_name) {
            expr = replace_expr;
        } else {
            expr = op;
        }
    }

private:
    std::string replced_var_name;
    Expr replace_expr;
};

Expr inverse(const Var &var, const Expr &expr) {
    // TODO: replace with a full visitor
    if (expr.get()->node_type == IRNodeType::Add) {
        const Add *op = expr.as<Add>();
        if (op->a.get()->node_type == IRNodeType::Variable &&
            op->a.as<Variable>()->name == var.name()) {
            return op->a - op->b; 
        } else {
            assert(op->b.get()->node_type == IRNodeType::Variable &&
                   op->b.as<Variable>()->name == var.name());
            return op->a - op->b; 
        }
    } else if (expr.get()->node_type == IRNodeType::Sub) {
        const Sub *op = expr.as<Sub>();
        if (op->a.get()->node_type == IRNodeType::Variable &&
            op->a.as<Variable>()->name == var.name()) {
            return op->a + op->b; 
        } else {
            assert(op->b.get()->node_type == IRNodeType::Variable &&
                   op->b.as<Variable>()->name == var.name());
            return op->b - op->a; 
        }
    } else if (expr.get()->node_type == IRNodeType::Max) {
        const Max *op = expr.as<Max>();
        if (op->a.get()->node_type == IRNodeType::IntImm) {
            return max(inverse(var, op->b), op->a); 
        } else {
            assert(op->b.get()->node_type == IRNodeType::IntImm);
            return max(inverse(var, op->a), op->b); 
        }
    } else if (expr.get()->node_type == IRNodeType::Min) {
        const Min *op = expr.as<Min>();
        if (op->a.get()->node_type == IRNodeType::IntImm) {
            return min(inverse(var, op->b), op->a); 
        } else {
            assert(op->b.get()->node_type == IRNodeType::IntImm);
            return min(inverse(var, op->a), op->b); 
        }
    } else if (expr.get()->node_type == IRNodeType::Variable) {
        return expr;
    }
    assert(false);
    return expr;
}

/** An IR graph visitor that gather the function DAG and sort them in reverse topological order
 */
class FunctionSorter : public IRGraphVisitor {
public:
    void sort(const Func &func);

    std::vector<Func> get_functions() const {
        return functions;
    }

    void visit(const Call *op);

private:
    std::vector<Func> functions;
    std::set<std::string> traversed_functions;
};

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

/** An IR visitor that computes the derivatives through reverse accumulation
 */
class ReverseAccumulationVisitor : public IRVisitor {
public:
    void propagate_adjoints(const std::vector<Func> &funcs);
    std::vector<Func> get_adjoint_funcs() const {
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
    void visit(const Call *op);
    void visit(const Let *op);
private:
    void accumulate(const Expr &stub, const Expr &adjoint);

    std::map<const BaseExprNode *, Expr> accumulated_adjoints;
    std::map<std::string, int> func_mapping;
    std::vector<Func> adjoint_funcs;
    std::map<std::string, Expr> let_var_mapping;
    std::vector<Var> current_args;
    OutputImageParam current_param;
};

void ReverseAccumulationVisitor::propagate_adjoints(const std::vector<Func> &funcs) {
    if (funcs.size() == 0) {
        return;
    }

    for (int i = 0; i < (int)funcs.size(); i++) {
        func_mapping[funcs[i].name()] = i;
        adjoint_funcs.push_back(Func(funcs[i].name() + "_d__"));
    }

    adjoint_funcs[0](funcs[0].args()) = 1.f;
    // Traverse functions
    for (int i = 0; i < (int)funcs.size(); i++) {
        const Func &func = funcs[i];
        current_args = func.args();
        // --- THIS DOES NOT WORK!!! --- //
        current_param = adjoint_funcs[i].output_buffer();
        ///////////////////////////////////

        // Traverse from the last update to first
        for (int update_id = func.num_update_definitions() - 1; update_id >= -1; update_id--) {
            // Topologically sort the expressions
            ExpressionSorter sorter;
            if (update_id >= 0) {
                sorter.sort(func.update_value(update_id));
            } else {
                sorter.sort(func.value());
            }

            std::vector<Expr> expr_list = sorter.get_expr_list();
            // Retrieve previously propagated adjoint
            if (update_id == func.num_update_definitions() - 1) {
                std::vector<Expr> args;
                for (const auto &arg : func.args()) {
                    args.push_back(arg);
                }
                accumulated_adjoints[(const BaseExprNode *)expr_list.back().get()] =
                    Call::make(adjoint_funcs[i].function(), args);
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

    // TODO: fix this
    // d/da min(a, b) = a <= b ? 1 : 0
    accumulate(op->a, adjoint);
    // d/db min(a, b) = b <= a ? 1 : 0
    accumulate(op->b, adjoint);
}

void ReverseAccumulationVisitor::visit(const Max *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    // TODO: fix this
    // d/da max(a, b) = a >= b ? 1 : 0
    accumulate(op->a, adjoint);
    // d/db max(a, b) = b >= a ? 1 : 0
    accumulate(op->b, adjoint);
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
        Func& func_to_update = adjoint_funcs[func_mapping[func.name()]];
        // Try to transform the scattering operation into gathering operation
        std::vector<Expr> inverse_args = op->args;
        assert(args.size() == inverse_args.size());
        VariableFinder var_finder;
        bool has_arg = true;
        for (int i = 0; i < (int)args.size(); i++) {
            if (var_finder.find(inverse_args[i], args[i])) {
                inverse_args[i] = inverse(args[i], inverse_args[i]);
            } else {
                has_arg = false;
            }
        }

        Func adjoint_func;
        adjoint_func(current_args) = 0.f;

        if (!has_arg) {
            std::vector<std::pair<Expr, Expr>> ranges;
            for (int i = 0; i < current_param.dimensions(); i++) {
                const Dimension &dim = current_param.dim(i);
                ranges.push_back(std::make_pair(dim.min(), dim.extent()));
            }
            RDom r(ranges);
            VariableReplacer var_replacer;
            for (int i = 0; i < (int)current_args.size(); i++) {
                adjoint = var_replacer.replace(adjoint, current_args[i].name(), r[i]);
            }
            for (int i = 0; i < (int)inverse_args.size(); i++) {
                if (inverse_args[i].as<Variable>() != nullptr) {
                    adjoint = var_replacer.replace(adjoint, inverse_args[i].as<Variable>()->name, args[i]);
                }
            }
        }
        debug(0) << "func.name():" << func.name() << "\n";
        debug(0) << "adjoint:" << adjoint << "\n";

        adjoint_func(current_args) += adjoint;
        Func new_func(func_to_update.name());
        if (func_to_update.defined()) {
            new_func(args) = 0.f;
            new_func(args) += func_to_update.value() + adjoint_func(inverse_args);
        } else {
            new_func(args) = 0.f;
            new_func(args) += adjoint_func(inverse_args);
        }
        adjoint_funcs[func_mapping[func.name()]] = new_func;
    }
}

void ReverseAccumulationVisitor::visit(const Let *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    accumulate(op->body, adjoint);
    let_var_mapping[op->name] = op->value;
}

} // namespace Internal

std::vector<Func> propagate_adjoints(const Func &output) {
    Internal::debug(0) << "propagate_adjoints begins" << "\n";
    Internal::FunctionSorter sorter;
    Internal::debug(0) << "sorting functions" << "\n";
    sorter.sort(output);
    std::vector<Func> funcs = sorter.get_functions();
    Internal::debug(0) << "func list:" << "\n";
    for (const auto &func : funcs) {
        Internal::debug(0) << func.name() << "\n";
    }
    Internal::ReverseAccumulationVisitor visitor;
    visitor.propagate_adjoints(funcs);
    return visitor.get_adjoint_funcs();
}

void print_func(const Func &func) {
    Internal::debug(0) << "printing function:" << func.name() << "\n";
    Internal::FunctionSorter sorter;
    sorter.sort(func);
    std::vector<Func> funcs = sorter.get_functions();
    for (int i = (int)funcs.size() - 1; i >= 0; i--) {
        Func &func = funcs[i];
        Internal::debug(0) << "funcs[" << i << "]:" << func.name() << "\n";
        for (int update_id = -1; update_id < func.num_update_definitions(); update_id++) {
            if (update_id >= 0) {
                Internal::debug(0) << "update:" << func.update_value(update_id) << "\n";
            } else {
                Internal::debug(0) << "init:" << func.value() << "\n";
            }
        }
    }
}

} // namespace Halide