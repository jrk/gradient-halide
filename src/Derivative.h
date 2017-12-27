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
#include <set>

namespace Halide {

// function name & update_id, for initialization update_id == -1
using FuncKey = std::pair<std::string, int>;

struct Derivative {
    std::map<FuncKey, Func> adjoints;
};

// Bounds are {min, max}
Derivative propagate_adjoints(const Func &output,
                              const Func &adjoint,
                              const std::vector<std::pair<Expr, Expr>> &output_bounds);
Derivative propagate_adjoints(const Func &output,
                              const Buffer<float> &adjoint);
Derivative propagate_adjoints(const Func &output);
void print_func(const Func &func, bool ignore_non_adjoints = true, bool ignore_bc = true, bool recursive = true, int depth = -1);
// Bounds are {min, max}
void simple_autoschedule(std::vector<Func> &outputs,
                         const std::map<std::string, int> &parameters,
                         const std::vector<std::vector<std::pair<int, int>>> &output_bounds,
                         const std::set<std::string> &skip_functions = {});
void simple_autoschedule(Func &output,
                         const std::map<std::string, int> &parameters,
                         const std::vector<std::pair<int, int>> &output_bounds,
                         const std::set<std::string> &skip_functions = {});

namespace Internal {

EXPORT void derivative_test();

}

}

#endif
