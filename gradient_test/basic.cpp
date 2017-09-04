// g++ basic.cpp -g -I ../include -L ../bin -lHalide -o basic -std=c++11
// DYLD_LIBRARY_PATH=../bin ./basic

#include "Halide.h"
#include <stdio.h>

constexpr float c_tolerance = 1e-6;

int main(int argc, char **argv) {
    Halide::Param<float> weight(1.25);
    Halide::Param<float> bias(-10);
    Halide::Var x, y;
    Halide::Expr output_expr = weight * (x + y) + bias;
    std::vector<Halide::Expr> derivative_exprs = Halide::derivative(output_expr, {weight, bias});
    // output = weight * (x + y) + bias
    // d_weight = (x + y)
    // d_bias = 1
    Halide::Func output_func, d_weight, d_bias;
    output_func(x, y) = output_expr;
    d_weight(x, y) = derivative_exprs[0];
    d_bias(x, y) = derivative_exprs[1];

    Halide::Buffer<float> output = output_func.realize(800, 600);
    for (int j = 0; j < output.height(); j++) {
        for (int i = 0; i < output.width(); i++) {
            const float ref = weight.get() * (i + j) + bias.get();
            if (fabs(output(i, j) - ref) > c_tolerance) {
                printf("Something went wrong!\n"
                       "Pixel %d, %d was supposed to be %f, but instead it's %f\n",
                       i, j, ref, output(i, j));
                return -1;
            }
        }
    }

    output = d_weight.realize(800, 600);
    for (int j = 0; j < output.height(); j++) {
        for (int i = 0; i < output.width(); i++) {
            const float ref = (i + j);
            if (fabs(output(i, j) - ref) > c_tolerance) {
                printf("Something went wrong!\n"
                       "Pixel %d, %d was supposed to be %f, but instead it's %f\n",
                       i, j, ref, output(i, j));
                return -1;
            }
        }
    }

    std::cerr << "Done" << std::endl;

    output = d_bias.realize(800, 600);
    for (int j = 0; j < output.height(); j++) {
        for (int i = 0; i < output.width(); i++) {
            const float ref = 1.f;
            if (fabs(output(i, j) - ref) > c_tolerance) {
                printf("Something went wrong!\n"
                       "Pixel %d, %d was supposed to be %f, but instead it's %f\n",
                       i, j, ref, output(i, j));
                return -1;
            }
        }
    }

    printf("Success!\n");
    return 0;
}
