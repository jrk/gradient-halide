#ifndef HALIDE_DERIVATIVE_H
#define HALIDE_DERIVATIVE_H

/** \file
 *  Given an Halide expression, compute the derivatives w.r.t. other Halide expressions.
 */

#include "Module.h"
#include "Expr.h"

#include <vector>

namespace Halide {

std::vector<Expr> derivative(Expr output, const std::vector<Expr> &arg_list);

}

#endif
