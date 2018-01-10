#ifndef HALIDE_INTERNAL_DERIVATIVE_UTILS_H
#define HALIDE_INTERNAL_DERIVATIVE_UTILS_H

#include "Expr.h"
#include "Var.h"
#include "RDom.h"
#include "Scope.h"
#include "Derivative.h"
#include "Bounds.h"

namespace Halide {
namespace Internal {

using FuncBounds = std::vector<std::pair<Expr, Expr>>;

bool has_variable(const Expr &expr, const std::string &name);
bool has_let_defined(const Expr &expr, const std::string &name);
Expr remove_let_definitions(const Expr &expr);
std::vector<std::string> gather_variables(const Expr &expr,
	const std::vector<std::string> &filter);
std::vector<std::string> gather_variables(const Expr &expr,
	const std::vector<Var> &filter);
std::map<std::string, std::pair<Expr, Expr>> gather_rvariables(Expr expr);
std::map<std::string, std::pair<Expr, Expr>> gather_rvariables(Tuple tuple);
Expr add_let_expression(const Expr &expr,
                        const std::map<std::string, Expr> &let_var_mapping,
                        const std::vector<std::string> &let_variables);
std::vector<Expr> sort_expressions(const Expr &expr);
std::map<std::string, Box> inference_bounds(const std::vector<Func> &funcs,
	 									    const std::vector<FuncBounds> &output_bounds);
std::map<std::string, Box> inference_bounds(const Func &func,
	 									    const FuncBounds &output_bounds);
std::vector<std::pair<Expr, Expr>> rdom_to_vector(const RDom &bounds);
std::vector<std::pair<Expr, Expr>> box_to_vector(const Box &bounds);
bool equal(const RDom &bounds0, const RDom &bounds1);
std::vector<std::string> vars_to_strings(const std::vector<Var> &vars);
ReductionDomain extract_rdom(const Expr &expr);
std::pair<bool, Expr> solve_inverse(Expr expr,
                                    const std::string &new_var,
                                    const std::string &var);
std::set<std::string> find_dependency(const Func &func);
bool is_pointwise(const Func &caller, const Func &callee);
std::map<std::string, int> find_buffers_dimensions(const Func &func);

}
}

#endif
