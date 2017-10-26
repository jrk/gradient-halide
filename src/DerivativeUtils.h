#ifndef HALIDE_INTERNAL_DERIVATIVE_UTILS_H
#define HALIDE_INTERNAL_DERIVATIVE_UTILS_H

#include "Expr.h"
#include "Var.h"
#include "RDom.h"
#include "Scope.h"
#include "Derivative.h"

namespace Halide {
namespace Internal {

using FuncBounds = std::vector<std::pair<Expr, Expr>>;

bool has_variable(const Expr &expr, const std::string &name);
bool has_let_defined(const Expr &expr, const std::string &name);
Expr remove_let_definitions(const Expr &expr);
std::vector<std::string> gather_variables(const Expr &expr,
	const std::vector<std::string> &filter);
std::map<std::string, std::pair<Expr, Expr>> gather_rvariables(const Expr &expr);
Expr add_let_expression(const Expr &expr,
                        const std::map<std::string, Expr> &let_var_mapping,
                        const std::vector<std::string> &let_variables);
std::vector<Expr> sort_expressions(const Expr &expr);
std::map<FuncKey, RDom> inference_bounds(const Func &func);
std::vector<std::pair<Expr, Expr>> rdom_to_vector(const RDom &bounds);

}
}

#endif