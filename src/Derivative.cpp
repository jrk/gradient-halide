#include "Derivative.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Error.h"
#include <iostream>

namespace Halide {
namespace Internal {

/** An IR mutator that mutates expression to obtain the derivatives
 */
class DerivativeMutator : public IRGraphMutator {
public:
    Expr derivative(const Expr &e, const std::string &arg_name);
    Expr mutate(const Expr &e);
protected:
    void visit(const Add *op);
    void visit(const Mul *op);
    void visit(const Variable *op);
private:
    std::string current_arg_name;
    std::vector<std::string> arg_name_list;
};

static int debug_indent = 0;

Expr DerivativeMutator::derivative(const Expr &e, const std::string &arg_name) {
    current_arg_name = arg_name;
    return mutate(e);
}

Expr DerivativeMutator::mutate(const Expr &e) {
    const std::string spaces(debug_indent, ' ');
    std::cerr << spaces << "Transforming Expr: " << e << "\n";

    auto iter = expr_replacements.find(e);
    if (iter != expr_replacements.end()) {
        Expr new_e = iter->second;
        if (!new_e.same_as(e)) {
            std::cerr
                << spaces << "Old Expr" << "\n"
                << spaces << "Before: " << e << "\n"
                << spaces << "After:  " << new_e << "\n";
        }

        return new_e;
    }
    debug_indent++;
    Expr new_e = IRMutator::mutate(e);
    debug_indent--;

    if (!new_e.same_as(e)) {
        std::cerr
            << spaces << "New Expr" << "\n"
            << spaces << "Before: " << e << "\n"
            << spaces << "After:  " << new_e << "\n";
    }

    if (e.as<Variable>() == nullptr) { // Never cache variable transform
        expr_replacements[e] = new_e;
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
    if (op->name == current_arg_name) {
        expr = 1;
    } else {
        expr = 0;
    }
}

} // namespace Internal

std::vector<Expr> derivative(Expr output, const std::vector<Expr> &arg_list) {
    Internal::DerivativeMutator derivative_mutator;
    std::vector<Expr> derivative_list;
    for (const auto &arg : arg_list) {
        // TODO: support non-variable expressions
        const Internal::Variable *var = arg.as<Internal::Variable>();
        if (var == nullptr) {
            throw CompileError("[derivative] argument needs to be a Variable");
        }

        derivative_list.push_back(derivative_mutator.derivative(output, var->name));
    }
    return derivative_list;
}

}