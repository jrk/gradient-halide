#include "Halide.h"

namespace {

using namespace Halide;

class DiffConvolutionLayer : public Halide::Generator<DiffConvolutionLayer> {
public:
    GeneratorParam<bool>  auto_schedule{"auto_schedule", false};

    Input<Buffer<float>>  input{"input", 4};
    Input<Buffer<float>>  filter{"filter", 4};
    Input<Buffer<float>>  bias{"bias", 1};
    Input<Buffer<float>>  compare{"compare", 4};

    Output<Buffer<float>> f_ReLU{"ReLU", 4};
    Output<Buffer<float>> d_filter{"d_filter", 4};
    Output<Buffer<float>> d_bias{"d_bias", 1};

    void generate() {
        /* THE ALGORITHM */

        Var x("x"), y("y"), z("z"), n("n");

        Func f_conv("conv");
        RDom r(filter.dim(0).min(), filter.dim(0).extent(),
               filter.dim(1).min(), filter.dim(1).extent(),
               filter.dim(2).min(), filter.dim(2).extent());
        Func f_filter("f_filter");
        Func f_bias("f_bias");
        f_filter(x, y, z, n) = filter(x, y, z, n);
        f_bias(z) = bias(z);

        f_conv(x, y, z, n) = 0.f;

        f_conv(x, y, z, n) += f_filter(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z, n);

        f_ReLU(x, y, z, n) = max(0, f_conv(x, y, z, n));

        RDom r_target(compare);
        Expr diff = f_ReLU(r_target.x, r_target.y, r_target.z, r_target.w) -
                    compare(r_target.x, r_target.y, r_target.z, r_target.w);
        Expr loss = diff * diff;
        std::map<std::string, Func> adjoints = propagate_adjoints(loss);
        d_filter(x, y, z, n) = adjoints[f_filter.name()](x, y, z, n);
        d_bias(z) = 0.f;//adjoints[f_bias.name()](z);

        /* THE SCHEDULE */

        if (auto_schedule) {
            // Provide estimates on the input image
            input.dim(0).set_bounds_estimate(0, 67);
            input.dim(1).set_bounds_estimate(0, 67);
            input.dim(2).set_bounds_estimate(0, 32);
            input.dim(3).set_bounds_estimate(0, 4);

            filter.dim(0).set_bounds_estimate(0, 3);
            filter.dim(1).set_bounds_estimate(0, 3);
            filter.dim(2).set_bounds_estimate(0, 32);
            filter.dim(3).set_bounds_estimate(0, 32);

            bias.dim(0).set_bounds_estimate(0, 32);

            compare.dim(0).set_bounds_estimate(0, 64);
            compare.dim(1).set_bounds_estimate(0, 64);
            compare.dim(2).set_bounds_estimate(0, 32);
            compare.dim(3).set_bounds_estimate(0, 4);

            // Provide estimates on the pipeline f_ReLU
            f_ReLU.estimate(x, 0, 128)
                    .estimate(y, 0, 128)
                    .estimate(z, 0, 64)
                    .estimate(n, 0, 4);

            d_filter.estimate(x, 0, 3)
                    .estimate(y, 0, 3)
                    .estimate(z, 0, 32)
                    .estimate(n, 0, 32);

            d_bias.estimate(z, 0, 32);

            // Auto schedule the pipeline: this calls auto_schedule() for
            // all of the Outputs in this Generator
            auto_schedule_outputs();

        } else {
            // Add compute_root() to forward pass
            f_conv.compute_root();
            f_ReLU.compute_root();
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(DiffConvolutionLayer, diff_conv_layer)
