#ifndef HALIDE_DERIVATIVE_H
#define HALIDE_DERIVATIVE_H

/** \file
 *  Propagate the adjoints of a Halide function
 */

#include "Module.h"
#include "Expr.h"
#include "Func.h"

#include <vector>

namespace Halide {

std::vector<Func> propagate_adjoints(const Func &output);

}

#endif
