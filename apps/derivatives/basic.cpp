#include "Halide.h"
#include "halide_image_io.h"

using Halide::Var;
using Halide::Expr;
using Halide::Buffer;
using Halide::Func;
using Halide::BoundaryConditions::repeat_edge;

int main(int argc, char **argv) {
    Var x("x"), y("y");
    Buffer<uint8_t> input = Halide::Tools::load_image("images/gray.png");


    return 0;
}
