#include "Halide.h"

using Halide::Func;
using Halide::Expr;
using Halide::Var;
using Halide::undef;
using Halide::BoundaryConditions::repeat_edge;

namespace hdrnet {

class SliceGenerator : public Halide::Generator<SliceGenerator> {
public:
  Input<Buffer<float>> input_grid{"input_grid", 4};
  Input<Buffer<float>> guide_image{"guide_image", 2};

  Var x, y, z, c, cxx, cxy, xi, xo, czz, xy, czxy;

  Pipeline build() {
    Func b_grid("b_grid");
    b_grid(x, y, z, c) = Halide::BoundaryConditions::repeat_edge(
        input_grid)(x, y, z, c);
    Func b_guide("b_guide");
    b_guide(x, y) = Halide::BoundaryConditions::repeat_edge(guide_image)(x, y);

    Expr gw = input_grid.dim(0).extent();
    Expr gh = input_grid.dim(1).extent();
    Expr gd = input_grid.dim(2).extent();
    Expr nc = input_grid.dim(3).extent();

    Expr w = guide_image.dim(0).extent();
    Expr h = guide_image.dim(1).extent();

    // Coordinates in the bilateral grid
    Expr gx = (x+0.5f)*gw/(1.0f*w);
    Expr gy = (y+0.5f)*gh/(1.0f*h);
    Expr gz = clamp(b_guide(x, y), 0.0f, 1.0f)*gd;

    // Floor voxel
    Expr fx = cast<int>(floor(gx-0.5f));
    Expr fy = cast<int>(floor(gy-0.5f));
    Expr fz = cast<int>(floor(gz-0.5f));

    // Ceil voxel
    Expr cx = fx+1;
    Expr cy = fy+1;
    Expr cz = fz+1;

    // weights
    Expr wx = gx-0.5f - fx;
    Expr wy = gy-0.5f - fy;
    Expr wz = gz-0.5f - fz;

    // Tri-linear interpolation
    Func interp_y("interp_y");
    Func interp_x("interp_x");
    Func output("output");

    interp_y(x, y, z, c) = b_grid(x, fy, z, c)*(1-wy) + b_grid(x, cy, z, c)*wy;
    interp_x(x, y, z, c) = interp_y(fx, y, z, c)*(1-wx) + interp_y(cx, y, z, c)*wx;
    output(x, y, c) = interp_x(x, y, fz, c)*(1-wz) + interp_x(x, y, cz, c)*wz;

    // CPU Schedule
    int parallel_sz = 2;
    int vector_w = 8;
    Expr vectorizable = output.output_buffer().dim(2).extent() >= vector_w;
    output
        .compute_root()
        .parallel(c, parallel_sz);
    output
        .specialize(vectorizable)
        .vectorize(x, vector_w);

    std::vector<Func> outputs = {
      output
    };
    return Pipeline(outputs);
  }
};


class SliceGradGenerator : public Halide::Generator<SliceGradGenerator> {
 public:
  GeneratorParam<Halide::Type> input_type{"input_type", Float(32)};

  Halide::ImageParam input_grid{input_type, 4, "input_grid"};
  Halide::ImageParam guide_image{input_type, 2, "guide_image"};
  Halide::ImageParam grad{input_type, 3, "grad"};

  Var x, y, z, c;
  Var xi, yi, xo, yo, zi, zo, xy, cz, czx, czy, czxy;

  Pipeline build() {
    x = Var("x");
    y = Var("y");
    c = Var("c");
    z = Var("z");
    xi = Var("xi");
    xo = Var("xo");
    yi = Var("yi");
    yo = Var("yo");
    zi = Var("zi");
    zo = Var("zo");
    xy = Var("xy");
    cz = Var("cz");
    czx = Var("czx");
    czy = Var("czy");
    czxy = Var("czxy");

    Func b_grad("b_grad");
    b_grad(c, x, y) =
        Halide::BoundaryConditions::constant_exterior(grad, 0.0f)(c, x, y);
    Func b_guide("b_guide");
    b_guide(x, y) = Halide::BoundaryConditions::repeat_edge(guide_image)(x, y);

    Expr gd = input_grid.extent(1);
    Expr gw = input_grid.extent(2);
    Expr gh = input_grid.extent(3);

    Expr w = guide_image.extent(0);
    Expr h = guide_image.extent(1);

    Func output("output");
    Func splatz("splatz");
    Func splatx("splatx");
    Func splaty("splaty");

    // Splat gradient according to guide
    Expr bg_z = clamp(b_guide(x, y), 0.0f, 1.0f)*(gd-1.0f);
    Expr wz = max(1.0f-abs(bg_z - z), 0.0f);
    splatz(c, z, x, y) = b_grad(c, x, y)*wz;

    Expr sw = w*1.0f/gw;
    Expr rsw = cast<int>(ceil(sw));
    RDom rx(0, 2*rsw+1, "rx");
    Expr left = cast<int>(floor(sw*((x-1)+0.5f)-0.5f));
    Expr ibg_x = left + rx;  // neighboor coord in out
    Expr bg_x = (ibg_x + 0.5f)*gw/(1.0f*w) - 0.5f;  // neighboor coord in bg
    Expr wx = max(1.0f-abs(bg_x-x), 0.0f);
    // splatx(c, z, x, y) = 0.0f;
    splatx(c, z, x, y) = sum(splatz(c, z, ibg_x , y)*wx);
    splatx(c, z, 0, y) += splatx(c, z, -1, y);
    splatx(c, z, gw-1, y) += splatx(c, z, gw, y);  // Boundary condition

    Expr sh = h*1.0f/gh;
    Expr rsh = cast<int>(ceil(sh));
    RDom ry(0, 2*rsh+1, "ry");
    Expr top = cast<int>(floor(sh*((y-1)+0.5f)-0.5f));
    Expr ibg_y = top + ry;  // neighboor coord in out
    Expr bg_y = (ibg_y + 0.5f)*gh/(1.0f*h) - 0.5f;  // neighboor coord in bg
    Expr wy = max(1.0f-abs(bg_y-y), 0.0f);
    // splaty(c, z, x, y) = 0.0f;
    splaty(c, z, x, y) = sum(splatx(c, z, x, ibg_y)*wy);
    // splaty(c, z, x, 0) += splaty(c, z, x, -1);
    // splaty(c, z, x, gh-1) += splaty(c, z, x, gh);

    output(c, z, x, y) = splaty(c, z, x, y);
    output(c, z, x, 0) += splaty(c, z, x, -1);
    output(c, z, x, gh-1) += splaty(c, z, x, gh);

    // CPU Schedule
    int parallel_sz = 2;
    int vector_w = 8;

    splatz
        .compute_inline();

    Expr vectorizable_x = output.output_buffer().extent(2) >= vector_w;
    Expr vectorizable_z = output.output_buffer().extent(1) >= vector_w;
    splatx
        .compute_root()
        .parallel(y, parallel_sz);
    splatx
        .update(0)
        .parallel(y, parallel_sz);
    splatx
        .update(1)
        .parallel(y, parallel_sz);

    splatx
        .specialize(vectorizable_x)
        .vectorize(x, vector_w);
    splatx
        .update(0)
        .specialize(vectorizable_z)
        .vectorize(z, vector_w);
    splatx
        .update(1)
        .specialize(vectorizable_z)
        .vectorize(z, vector_w);

    splaty
        .compute_inline();

    Expr vectorizable = output.output_buffer().extent(2) >= vector_w;
    vectorizable_z = output.output_buffer().extent(1) >= vector_w;

    output
        .compute_root()
        .parallel(y, parallel_sz);
    output
        .update(0)
        .parallel(x, parallel_sz);
    output
        .update(1)
        .parallel(x, parallel_sz);

    output
        .specialize(vectorizable)
        .vectorize(x, vector_w);
    output
        .update(0)
        .specialize(vectorizable_z)
        .vectorize(z, vector_w);
    output
        .update(1)
        .specialize(vectorizable_z)
        .vectorize(z, vector_w);

    std::vector<Func> outputs = {
      output
    };
    return Pipeline(outputs);
  }
};

}  // end namespace hdrnet

HALIDE_REGISTER_GENERATOR(hdrnet::SliceGenerator, hdrnet_slice)
// HALIDE_REGISTER_GENERATOR(hdrnet::SliceGradGenerator, hdrnet_slice_grad)

