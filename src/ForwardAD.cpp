#include "ForwardAD.h"
#include "IROperator.h"
#include "Error.h"
#include <iostream>

namespace Halide {
namespace Internal {

/** A mutator that mutates expression using forward automatic differentiation
 */
class ForwardADMutator : public IRMutator {
public:
	ForwardADMutator(const Expr &wrtExpr);
	Expr mutate(const Expr &e);
protected:
	void visit(const Add *op);
	void visit(const Mul *op);
	void visit(const Variable *op);
private:
	const Variable *wrt;
};

static int debug_indent = 0;

ForwardADMutator::ForwardADMutator(const Expr &wrtExpr) {
	wrt = wrtExpr.as<Variable>();
	if (wrt == nullptr) {
		throw CompileError("[ForwardADMutator] wrt needs to be a Variable");
	}
}

Expr ForwardADMutator::mutate(const Expr &e) {
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

void ForwardADMutator::visit(const Add *op) {
    Expr da = mutate(op->a);
    Expr db = mutate(op->b);
    // d/dx a + b = da/dx + db/dx
    expr = da + db;
}

void ForwardADMutator::visit(const Mul *op) {
    Expr da = mutate(op->a);
    Expr db = mutate(op->b);
    // d/dx a * b = da/dx * b + a * db/dx
    expr = da * op->b + op->a * db;
}

void ForwardADMutator::visit(const Variable *op) {
    if (op->name == wrt->name) {
    	expr = 1;
	} else {
		expr = 0;
	}
}

} // namespace Internal

Expr forward_ad(Expr output, Expr wrt) {
	Internal::ForwardADMutator forward_ad_mutator(wrt);
	return forward_ad_mutator.mutate(output);
}

}