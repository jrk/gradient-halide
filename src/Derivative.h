#ifndef HALIDE_DERIVATIVE_H
#define HALIDE_DERIVATIVE_H

/** \file
 *  Automatic differentiation
 */

#include "Module.h"
#include "Expr.h"
#include "Func.h"

#include <vector>
#include <array>

namespace Halide {

// function name & update_id, for initialization update_id == -1
using FuncKey = std::pair<std::string, int>;

struct Derivative {
    std::map<FuncKey, Func> adjoints;
    std::map<FuncKey, RDom> reductions;
};

// Bounds are {min, max}
Derivative propagate_adjoints(const Func &output,
                              const Func &adjoint,
                              const std::vector<std::pair<Expr, Expr>> &output_bounds);
Derivative propagate_adjoints(const Func &output,
                              const Buffer<float> &adjoint);
Derivative propagate_adjoints(const Func &output);
void print_func(const Func &func, bool recursive = true);

namespace Internal {

EXPORT void derivative_test();

}

}

#endif
