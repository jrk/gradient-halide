#include <cstdio>
#include <chrono>

#include "diff_conv_layer.h"
#include "diff_conv_layer_auto_schedule.h"

#include "halide_benchmark.h"
#include "HalideBuffer.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    Buffer<float> input(67, 67, 32, 4);
    Buffer<float> filter(3, 3, 32, 32);
    Buffer<float> bias(32);
    Buffer<float> target(64, 64, 32, 4);

    for (int c = 0; c < input.dim(3).extent(); c++) {
        for (int z = 0; z < input.channels(); z++) {
            for (int y = 0; y < input.height(); y++) {
                for (int x = 0; x < input.width(); x++) {
                    input(x, y) = rand();
                }
            }
        }
    }

    for (int c = 0; c < filter.dim(3).extent(); c++) {
        for (int z = 0; z < filter.channels(); z++) {
            for (int y = 0; y < filter.height(); y++) {
                for (int x = 0; x < filter.width(); x++) {
                    filter(x, y) = rand();
                }
            }
        }
    }

    for (int x = 0; x < bias.width(); x++) {
        bias(x) = rand();
    }

    for (int c = 0; c < target.dim(3).extent(); c++) {
        for (int z = 0; z < target.channels(); z++) {
            for (int y = 0; y < target.height(); y++) {
                for (int x = 0; x < target.width(); x++) {
                    target(x, y) = rand();
                }
            }
        }
    }

    Buffer<float> output(64, 64, 32, 4);
    Buffer<float> d_filter(3, 3, 32, 32);
    Buffer<float> d_bias(32);

    diff_conv_layer(input, filter, bias, target, output, d_filter);

    // Timing code

    // Manually-tuned version
    double min_t_manual = benchmark(10, 10, [&]() {
        diff_conv_layer(input, filter, bias, target, output, d_filter);
    });
    printf("Manually-tuned time: %gms\n", min_t_manual * 1e3);

    // Auto-scheduled version
    double min_t_auto = benchmark(10, 10, [&]() {
        diff_conv_layer_auto_schedule(input, filter, bias, target, output, d_filter);
    });
    printf("Auto-scheduled time: %gms\n", min_t_auto * 1e3);

    return 0;
}
