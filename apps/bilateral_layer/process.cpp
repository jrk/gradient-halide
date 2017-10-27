#include <cstdio>
#include <chrono>

#include "bilateral_layer.h"

#include "halide_benchmark.h"
#include "HalideBuffer.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    Buffer<float> input(67, 67);
    Buffer<float> filter(3, 3, 8);
    Buffer<float> bias(8);
    Buffer<float> target(64, 64);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = rand();
        }
    }

    for (int z = 0; z < filter.channels(); z++) {
        for (int y = 0; y < filter.height(); y++) {
            for (int x = 0; x < filter.width(); x++) {
                filter(x, y, z) = rand();
            }
        }
    }

    for (int x = 0; x < bias.width(); x++) {
        bias(x) = rand();
    }

    for (int y = 0; y < target.height(); y++) {
        for (int x = 0; x < target.width(); x++) {
            target(x, y) = rand();
        }
    }

    Buffer<float> output(64, 64);
    Buffer<float> d_filter(3, 3, 8);
    Buffer<float> d_bias(8);

    bilateral_layer(input, filter, bias, target, output, d_filter, d_bias);

    // Timing code

    // Auto-scheduled version
    double min_t_auto = benchmark(1, 1, [&]() {
        bilateral_layer(input, filter, bias, target, output, d_filter, d_bias);
    });
    printf("Time: %gms\n", min_t_auto * 1e3);

    return 0;
}
