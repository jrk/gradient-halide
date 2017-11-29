#include "Halide.h"

namespace {

using namespace Halide;

class BilateralLayer : public Halide::Generator<BilateralLayer> {
public:
    Input<Buffer<float>>  input{"input", 4};       // x, y, channel, batch size
    Input<Buffer<float>>  guide{"guide", 3};       // x, y, batch size
    Input<Buffer<float>>  filter{"filter", 5};     // x, y, z offset, input channel, output channel
    Input<Buffer<float>>  bias{"bias", 2};         // z offset, channel
    Input<Buffer<float>>  adjoint{"adjoint", 4};   // x, y, channel, batch size

    Output<Buffer<float>> output{"output", 4};     // x, y, channel, batch size
    Output<Buffer<float>> d_input{"d_input", 4};   // same as input
    Output<Buffer<float>> d_guide{"d_guide", 3};   // same as guide
    Output<Buffer<float>> d_filter{"d_filter", 5}; // same as filter
    Output<Buffer<float>> d_bias{"d_bias", 2};     // same as bias

    void generate() {
        /* THE ALGORITHM */

        Var x("x"), y("y"), z("z"), n("n"), ci("ci"), co("co");

        // TODO: should we downsample the input here?

        // Offsets the input by different biases and applies ReLU
        // Do this for each channel
        Func f_input("input");
        f_input(x, y, ci, n) = input(x, y, ci, n);
        Func f_guide("guide");
        f_guide(x, y, n) = guide(x, y, n);
        Func f_bias("bias");
        f_bias(z, ci) = bias(z, ci);
        Func f_offset("offset");
        // x, y, z offset, input channel, batch size
        f_offset(x, y, z, ci, n) = max(0.f, f_input(x, y, ci, n) + f_bias(z, ci));
        // Perform 3D filtering in the offseted space
        // Again, do this for each channel
        // We assume the z offset part is fully-connected 
        // (i.e. filter.dim(2).extent() == bias.dim(1).extent())
        RDom r(filter.dim(0).min(), filter.dim(0).extent(),  // x
               filter.dim(1).min(), filter.dim(1).extent(),  // y
               filter.dim(2).min(), filter.dim(2).extent(),  // z offset
               filter.dim(3).min(), filter.dim(3).extent()); // input channel
        Func f_filter("filter");
        f_filter(x, y, z, ci, co) = filter(x, y, z, ci, co);
        Func f_conv("conv");
        f_conv(x, y, z, co, n)  = 0.f;
        f_conv(x, y, z, co, n) += f_filter(r[0], r[1], r[2], r[3], co) *
                                  f_offset(x + r[0], y + r[1], r[2], r[3], n);
        // Slice the result back to 2D
        // Find the coordinate in z
        Expr gz = clamp(f_guide(x, y, n), 0.0f, 1.0f) * (filter.dim(2).extent() - 1);
        // Floor voxel
        Expr fz = cast<int>(floor(gz) - 0.5f);
        // Ceil voxel
        Expr cz = fz + 1;
        // Weight
        Expr wz = gz - fz;
        // Linear interpolation
        Func f_output("output");
        f_output(x, y, co, n) = f_conv(x, y, fz, co, n) * (1.f - wz) +
                                f_conv(x, y, cz, co, n) * wz;

        Derivative d = propagate_adjoints(f_output, adjoint,
                                          {{adjoint.dim(0).min(), adjoint.dim(0).max()},
                                           {adjoint.dim(1).min(), adjoint.dim(1).max()},
                                           {adjoint.dim(2).min(), adjoint.dim(2).max()},
                                           {adjoint.dim(3).min(), adjoint.dim(3).max()}});
        std::map<FuncKey, Func> adjoints = d.adjoints;
        Func f_d_input = adjoints[FuncKey{f_input.name(), -1}];
        Func f_d_guide = adjoints[FuncKey{f_guide.name(), -1}];
        Func f_d_filter = adjoints[FuncKey{f_filter.name(), -1}];
        Func f_d_bias = adjoints[FuncKey{f_bias.name(), -1}];

        output(x, y, co, n) = f_output(x, y, co, n);
        d_input(x, y, ci, n) = f_d_input(x, y, ci, n);
        d_guide(x, y, n) = f_d_guide(x, y, n);
        d_filter(x, y, z, ci, co) = f_d_filter(x, y, z, ci, co);
        d_bias(z, ci) = f_d_bias(z, ci);

        /* THE SCHEDULE */
        if (auto_schedule) {
            // Provide estimates on the input image
            input.dim(0).set_bounds_estimate(0, 67);
            input.dim(1).set_bounds_estimate(0, 67);
            input.dim(2).set_bounds_estimate(0, 1);
            input.dim(3).set_bounds_estimate(0, 1);

            filter.dim(0).set_bounds_estimate(0, 3);
            filter.dim(1).set_bounds_estimate(0, 3);
            filter.dim(2).set_bounds_estimate(0, 8);
            filter.dim(3).set_bounds_estimate(0, 1);
            filter.dim(4).set_bounds_estimate(0, 1);

            bias.dim(0).set_bounds_estimate(0, 8);
            bias.dim(1).set_bounds_estimate(0, 1);

            adjoint.dim(0).set_bounds_estimate(0, 64);
            adjoint.dim(1).set_bounds_estimate(0, 64);
            adjoint.dim(2).set_bounds_estimate(0, 1);
            adjoint.dim(3).set_bounds_estimate(0, 1);

            // Provide estimates on the pipeline f_ReLU
            output.estimate(x , 0, 64)
                  .estimate(y , 0, 64)
                  .estimate(co, 0, 1)
                  .estimate(n , 0, 1);

            d_input.estimate(x , 0, 67)
                   .estimate(y , 0, 67)
                   .estimate(ci, 0, 1)
                   .estimate(n , 0, 1);

            d_guide.estimate(x, 0, 67)
                   .estimate(y, 0, 67)
                   .estimate(n, 0, 1);

            d_filter.estimate(x , 0, 3)
                    .estimate(y , 0, 3)
                    .estimate(z , 0, 8)
                    .estimate(ci, 0, 1)
                    .estimate(co, 0, 1);

            d_bias.estimate(z , 0, 1)
                  .estimate(ci, 0, 1);
        } else {
            f_offset.compute_root();
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
