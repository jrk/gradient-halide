#include <cstdio>
#include <chrono>

#include "bilateral_layer.h"

#include "halide_benchmark.h"
#include "HalideBuffer.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    Buffer<float> input(67, 67, 1, 1);
    Buffer<float> guide(67, 67, 1);
    Buffer<float> filter(3, 3, 8, 1, 1);
    Buffer<float> bias(8, 1);
    Buffer<float> adjoint(64, 64, 1, 1);

    for (int n = 0; n < input.dim(3).extent(); n++) {
        for (int c = 0; c < input.channels(); c++) {
            for (int y = 0; y < input.height(); y++) {
                for (int x = 0; x < input.width(); x++) {
                    input(x, y, c, n) = rand();
                }
            }
        }
    }

    for (int n = 0; n < input.channels(); n++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
                guide(x, y, n) = rand();
            }
        }
    }

    for (int co = 0; co < filter.dim(4).extent(); co++) {
        for (int ci = 0; ci < filter.dim(3).extent(); ci++) {
            for (int z = 0; z < filter.channels(); z++) {
                for (int y = 0; y < filter.height(); y++) {
                    for (int x = 0; x < filter.width(); x++) {
                        filter(x, y, z, ci, co) = rand();
                    }
                }
            }
        }
    }

    for (int c = 0; c < bias.height(); c++) {
        for (int z = 0; z < bias.width(); z++) {
            bias(z, c) = rand();
        }
    }

    for (int n = 0; n < adjoint.dim(3).extent(); n++) {
        for (int c = 0; c < adjoint.channels(); c++) {
            for (int y = 0; y < adjoint.height(); y++) {
                for (int x = 0; x < adjoint.width(); x++) {
                    adjoint(x, y, c, n) = rand();
                }
            }
        }
    }

    Buffer<float> output(64, 64, 1, 1);
    Buffer<float> d_input(67, 67, 1, 1);
    Buffer<float> d_guide(67, 67, 1);
    Buffer<float> d_filter(3, 3, 8, 1, 1);
    Buffer<float> d_bias(8, 1);

    bilateral_layer(input, guide, filter, bias, adjoint, output, d_input, d_guide, d_filter, d_bias);

    // Timing code

    // Auto-scheduled version
    double min_t_auto = benchmark(1, 1, [&]() {
        bilateral_layer(input, guide, filter, bias, adjoint, output, d_input, d_guide, d_filter, d_bias);
    });
    printf("Time: %gms\n", min_t_auto * 1e3);

    return 0;
}
