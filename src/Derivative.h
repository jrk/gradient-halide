#ifndef HALIDE_DERIVATIVE_H
#define HALIDE_DERIVATIVE_H

/** \file
 *  Given an Halide expression, compute the derivatives w.r.t. other Halide expressions.
 */

#include <ostream>

#include "Module.h"

namespace Halide {

Expr derivative(Expr output, Expr wrt);

}

#endif
