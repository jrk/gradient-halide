#ifndef HALIDE_DERIVATIVE_H
#define HALIDE_DERIVATIVE_H

/** \file
 *  Automatic differentiation
 */

#include "Module.h"
#include "Expr.h"
#include "Func.h"

#include <vector>

namespace Halide {

// function name & update_id, for initialization update_id == -1
using FuncKey = std::pair<std::string, int>;

struct Derivative {
    std::map<std::string, Func> adjoints;
    std::map<FuncKey, RDom> reductions;
};

Derivative propagate_adjoints(const Expr &output);
void print_func(const Func &func);

namespace Internal {

EXPORT void derivative_test();

}

}

#endif
