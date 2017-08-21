#ifndef HALIDE_FORWARD_AD_H
#define HALIDE_FORWARD_AD_H

/** \file
 * An IR mutator that transforms an Expression using forward automatic differentiation
 */

#include <ostream>

#include "Module.h"
#include "IRMutator.h"

namespace Halide {

Expr forward_ad(Expr output, Expr wrt);

}

#endif
