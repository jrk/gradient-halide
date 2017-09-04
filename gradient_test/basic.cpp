// g++ basic.cpp -g -I ../include -L ../bin -lHalide -o basic -std=c++11
// DYLD_LIBRARY_PATH=../bin ./basic

// The only Halide header file you need is Halide.h. It includes all of Halide.
#include "Halide.h"

// We'll also include stdio for printf.
#include <stdio.h>

int main(int argc, char **argv) {
    Halide::Param<float> factor(1.333);
    Halide::Func gradient;
    Halide::Var x, y;
    std::vector<Halide::Expr> e = Halide::derivative(factor * (x + y), {factor});
    gradient(x, y) = e[0];
    Halide::Buffer<float> output = gradient.realize(800, 600);
    for (int j = 0; j < output.height(); j++) {
        for (int i = 0; i < output.width(); i++) {
            if (output(i, j) != (i + j)) {
                printf("Something went wrong!\n"
                       "Pixel %d, %d was supposed to be %f, but instead it's %f\n",
                       i, j, float(i + j), output(i, j));
                return -1;
            }
        }
    }
    printf("Success!\n");
    return 0;
}
