#include "Derivative.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Error.h"
#include <iostream>

namespace Halide {
namespace Internal {

/** An IR mutator that mutates expression to obtain the derivatives
 */
class DerivativeMutator : public IRMutator {
public:
    DerivativeMutator(const Expr &wrtExpr);
    Expr mutate(const Expr &e);
protected:
    void visit(const Add *op);
    void visit(const Mul *op);
    void visit(const Variable *op);
private:
    const Variable *wrt;
};

static int debug_indent = 0;

DerivativeMutator::DerivativeMutator(const Expr &wrtExpr) {
    wrt = wrtExpr.as<Variable>();
    if (wrt == nullptr) {
        throw CompileError("[DerivativeMutator] wrt needs to be a Variable");
    }
}

Expr DerivativeMutator::mutate(const Expr &e) {
    const std::string spaces(debug_indent, ' ');
    std::cerr << spaces << "Transforming Expr: " << e << "\n";
    debug_indent++;
    Expr new_e = IRMutator::mutate(e);
    debug_indent--;
    if (!new_e.same_as(e)) {
        std::cerr
            << spaces << "Before: " << e << "\n"
            << spaces << "After:  " << new_e << "\n";
    }
    return new_e;
}

void DerivativeMutator::visit(const Add *op) {
    Expr da = mutate(op->a);
    Expr db = mutate(op->b);
    // d/dx a + b = da/dx + db/dx
    expr = da + db;
}

void DerivativeMutator::visit(const Mul *op) {
    Expr da = mutate(op->a);
    Expr db = mutate(op->b);
    // d/dx a * b = da/dx * b + a * db/dx
    expr = da * op->b + op->a * db;
}

void DerivativeMutator::visit(const Variable *op) {
    if (op->name == wrt->name) {
        expr = 1;
    } else {
        expr = 0;
    }
}

} // namespace Internal

Expr derivative(Expr output, Expr wrt) {
    Internal::DerivativeMutator derivative_mutator(wrt);
    return derivative_mutator.mutate(output);
}

}