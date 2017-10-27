#include "Halide.h"

namespace {

using namespace Halide;

class BilateralLayer : public Halide::Generator<BilateralLayer> {
public:
    Input<Buffer<float>>  input{"input", 2};
    Input<Buffer<float>>  filter{"filter", 3};
    Input<Buffer<float>>  bias{"bias", 1};
    Input<Buffer<float>>  compare{"compare", 2};

    Output<Buffer<float>> output{"output", 2};
    Output<Buffer<float>> d_filter{"d_filter", 3};
    Output<Buffer<float>> d_bias{"d_bias", 1};

    void generate() {
        /* THE ALGORITHM */

        Var x("x"), y("y"), z("z"), n("n");

        // TODO: should we downsample the input here?
        // TODO: get rid of clamping
        Func f_input("f_input");
        Expr clamped_x = Halide::clamp(x, 0, input.width() - 1);
        Expr clamped_y = Halide::clamp(y, 0, input.height() - 1);
        f_input(x, y) = input(clamped_x, clamped_y);

        // Offsets the input by different biases and applies ReLU
        Func f_bias("bias");
        f_bias(x) = bias(x);
        Func f_offset("offset");
        f_offset(x, y, z) = max(0.f, f_input(x, y) + f_bias(z));
        // Perform 3D filtering in the offseted space
        RDom r(filter.dim(0).min(), filter.dim(0).extent(),
               filter.dim(1).min(), filter.dim(1).extent(),
               filter.dim(2).min(), filter.dim(2).extent());
        Func f_filter("filter");
        f_filter(x, y, z) = filter(x, y, z);
        Func f_conv("conv");
        f_conv(x, y, z)  = 0.f;
        f_conv(x, y, z) += f_filter(r.x, r.y, r.z) * f_offset(x + r.x, y + r.y, r.z);
        // Slice the result back to 2D
        // Find the coordinate in z
        Expr gz = clamp(f_input(x, y), 0.0f, 1.0f) * (filter.dim(2).extent() - 1);
        // Floor voxel
        Expr fz = cast<int>(floor(gz) - 0.5f);
        // Ceil voxel
        Expr cz = fz + 1;
        // Weight
        Expr wz = gz - fz;
        // Linear interpolation
        Func f_output("output");
        f_output(x, y) = f_conv(x, y, fz) * (1.f - wz) + f_conv(x, y, cz) * wz;

        // RDom r_output(compare);
        RDom r_output(0, 64, 0, 64);
        Func f_loss("loss");
        f_loss(x) = 0.f;
        f_loss(x) += f_output(r_output.x, r_output.y);
        Derivative d = propagate_adjoints(f_loss);
        std::map<FuncKey, Func> adjoints = d.adjoints;
        Func f_d_filter = adjoints[FuncKey{f_filter.name(), -1}];
        Func f_d_bias = adjoints[FuncKey{f_bias.name(), -1}];
        print_func(f_d_bias);

        output(x, y) = f_output(x, y);
        d_filter(x, y, z) = f_d_filter(x, y, z);
        d_bias(x) = f_d_bias(x);

        /* THE SCHEDULE */
        if (auto_schedule) {
            // Provide estimates on the input image
            input.dim(0).set_bounds_estimate(0, 67);
            input.dim(1).set_bounds_estimate(0, 67);

            filter.dim(0).set_bounds_estimate(0, 3);
            filter.dim(1).set_bounds_estimate(0, 3);
            filter.dim(2).set_bounds_estimate(0, 8);

            bias.dim(0).set_bounds_estimate(0, 8);

            compare.dim(0).set_bounds_estimate(0, 64);
            compare.dim(1).set_bounds_estimate(0, 64);

            // Provide estimates on the pipeline f_ReLU
            output.estimate(x, 0, 64)
                  .estimate(y, 0, 64);

            d_filter.estimate(x, 0, 3)
                    .estimate(y, 0, 3)
                    .estimate(z, 0, 8);

            d_bias.estimate(x, 0, 1);
        } else {
            f_output.compute_root();
            f_conv.compute_root();
            adjoints[FuncKey{f_output.name(), -1}].compute_root();
            adjoints[FuncKey{f_conv.name(), 0}].compute_root();
            adjoints[FuncKey{f_offset.name(), 0}].compute_root();
            adjoints[FuncKey{f_d_bias.name(), 0}].compute_root();
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(BilateralLayer, bilateral_layer)
