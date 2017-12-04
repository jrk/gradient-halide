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
    //Output<Buffer<float>> d_bias{"d_bias", 1};

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

        f_conv(x, y, z, n) = f_bias(z);
        f_conv(x, y, z, n) += f_filter(r.x, r.y, r.z, z) * input(x + r.x, y + r.y, r.z, n);
        f_ReLU(x, y, z, n) = max(0, f_conv(x, y, z, n));

        RDom r_target(compare);
        Expr loss = f_ReLU(r_target.x, r_target.y, r_target.z, r_target.w);
        Derivative derivative = propagate_adjoints(loss);
        std::map<FuncKey, Func> adjoints = derivative.adjoints;
        d_filter(x, y, z, n) = adjoints[FuncKey{f_filter.name(), -1}](x, y, z, n);
        //d_bias(z) = 0.f;//adjoints[FuncKey{f_bias.name(), -1}](z);

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
            f_ReLU.estimate(x, 0, 64)
                    .estimate(y, 0, 64)
                    .estimate(z, 0, 32)
                    .estimate(n, 0, 4);

            d_filter.estimate(x, 0, 3)
                    .estimate(y, 0, 3)
                    .estimate(z, 0, 32)
                    .estimate(n, 0, 32);

            //d_bias.estimate(z, 0, 32);

            // Auto schedule the pipeline: this calls auto_schedule() for
            // all of the Outputs in this Generator
            auto_schedule_outputs();

            d_filter.print_loop_nest();
        } else {
            f_conv.compute_root();
            f_conv
              .parallel(n)
              .parallel(z)
              .vectorize(x, 8);

            f_conv.update()
                  .reorder(r.x, r.y, x, y, z, r.z, n)
                  .parallel(n)
                  .parallel(z)
                  .vectorize(x, 8);
            f_ReLU.compute_root();
            f_ReLU.parallel(n)
                  .parallel(z)
                  .vectorize(x, 8);

            Func &d_conv_1 = adjoints[FuncKey{f_conv.name(), 0}];
            d_conv_1.compute_root();
            /*d_conv_1.parallel(n)
                    .parallel(z)
                    .vectorize(x, 8);*/
            d_conv_1.update(0)
                    .parallel(n)
                    .parallel(z)
                    .vectorize(x, 8);

            Func &d_ReLU = adjoints[FuncKey{f_ReLU.name(), -1}];
            d_ReLU.compute_root();
            d_ReLU.parallel(n)
                  .parallel(z)
                  .vectorize(x, 8);
            d_ReLU.update(0)
                  .parallel(n)
                  .parallel(z)
                  .vectorize(x, 8);
            d_ReLU.update(1)
                  .parallel(n)
                  .parallel(z)
                  .vectorize(x, 8);

            Func &d_filter = adjoints[FuncKey{f_filter.name(), -1}];
            RDom r_conv = derivative.reductions[{d_filter.name(), 0}];
            d_filter.compute_root();
            d_filter.update(0)
                    .parallel(n)
                    .parallel(z);
            d_filter.update(1)
                    .parallel(n)
                    .parallel(z);
            Var rx("rx");
            Func intermediate = d_filter.update(0).rfactor(r_conv.x, rx);
            intermediate.compute_at(d_filter, x);
            intermediate.update(0)
                        .vectorize(rx, 8);
            d_filter.print_loop_nest();
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(DiffConvolutionLayer, diff_conv_layer)
