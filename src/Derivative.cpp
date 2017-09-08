#include "Derivative.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Error.h"
#include <iostream>

namespace Halide {
namespace Internal {

/** An IR graph visitor that gather the expression DAG and sort them in topological order
 */
class ExpressionSorter : public IRGraphVisitor {
public:
    void sort(const Expr &expr);

    std::vector<Expr> get_expr_list() const {
        return expr_list;
    }

protected:
    void include(const Expr &e);
private:
    std::vector<Expr> expr_list;
};

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

void ExpressionSorter::sort(const Expr &e) {
    expr_list.clear();
    e.accept(this);
    expr_list.push_back(e);
}

/** An IR visitor that computes the derivatives through reverse accumulation
 */
class ReverseAccumulationVisitor : public IRVisitor {
public:
    std::vector<Expr> derivative(const Expr &e, const std::vector<Expr> &arg_list);
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
private:
    void accumulate(const Expr &stub, const Expr &adjoint);

    std::vector<Expr> results;
    std::map<std::string, size_t> arg_map;
    std::map<const BaseExprNode *, Expr> accumulated_adjoints;
};

std::vector<Expr> ReverseAccumulationVisitor::derivative(const Expr &e, const std::vector<Expr> &arg_list) {
    // Preallocate the results and argument maps
    for (size_t arg_id = 0; arg_id < arg_list.size(); arg_id++) {
        results.push_back(0);
        // TODO: support non-variable expressions
        const Variable *var = arg_list[arg_id].as<Variable>();
        if (var == nullptr) {
            throw CompileError("[ReverseAccumulationVisitor] argument needs to be a Variable");
        }
        arg_map[var->name] = arg_id;
    }

    // Topologically sort the expressions
    ExpressionSorter sorter;
    sorter.sort(e);
    std::vector<Expr> expr_list = sorter.get_expr_list();
    accumulated_adjoints[(const BaseExprNode *)expr_list.back().get()] = Expr(1.f);
    // Traverse the expressions in reverse order
    for (auto it = expr_list.rbegin(); it != expr_list.rend(); it++) {
        // Propagate adjoints
        it->accept(this);
    }

    return results;
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
    if (arg_map.find(op->name) != arg_map.end()) {
        results[arg_map[op->name]] = accumulated_adjoints[op];
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

    // d/da min(a, b) = 1
    accumulate(op->a, adjoint);
    // d/db min(a, b) = 1
    accumulate(op->b, adjoint);
}

void ReverseAccumulationVisitor::visit(const Max *op) {
    assert(accumulated_adjoints.find(op) != accumulated_adjoints.end());
    Expr adjoint = accumulated_adjoints[op];

    // d/da max(a, b) = 1
    accumulate(op->a, adjoint);
    // d/db max(a, b) = 1
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
        // TODO: propagate through Functions
        for (size_t i = 0; i < op->args.size(); i++) {
            accumulate(op->args[i], adjoint);
        }
    } else if (op->image.defined()) {

    }
}

} // namespace Internal

std::vector<Expr> derivative(Expr output, const std::vector<Expr> &arg_list) {
    Internal::ReverseAccumulationVisitor visitor;
    return visitor.derivative(output, arg_list);
}

} // namespace Halide