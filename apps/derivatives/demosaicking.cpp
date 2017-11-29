#include "Halide.h"
#include "halide_image_io.h"

using Halide::Var;
using Halide::Expr;
using Halide::Buffer;
using Halide::Func;
using Halide::BoundaryConditions::repeat_edge;

int main(int argc, char **argv) {
    Var x("x"), y("y");
    Buffer<uint8_t> input = Halide::Tools::load_image("images/signs-small.png");
    std::cout << "Loaded image with size " << input.width() << "x" << input.height() << std::endl;

    Func input_float("input_float");
    input_float(x, y) = Halide::cast<float>(input(x, y))/255.0f;

    Func clamped("clamped");
    clamped(x, y) = repeat_edge(input)(x, y);

    Func weight_param("weight_param");

    Func even_row_g("even_row_g");
    even_row_g(x, y) = clamped(2 * x, 2 * y);
    Func even_row_r("even_row_r");
    even_row_r(x, y) = clamped(2 * x + 1, 2 * y);
    Func even_row_interpolated_g("even_row_interpolated_g");
    even_row_g(x, y) =  

    Func blur_x("blur_x");
    blur_x(x, y) = (input_float(x, y) + input_float(x+1, y) + input_float(x+2, y))/3.0f;

    Func blur_y("blur_y");
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2))/3.0f;

    Halide::RDom r(0, input.width()-2, 0, input.height()-2);  // No proper boundary conditions yet
    Expr diff = blur_y(r.x, r.y) - input_float(r.x, r.y);
    Expr loss = diff * diff;

    Func loss_f("loss_f");
    loss_f(x) = 0.0f;
    loss_f(x) += loss;
    // loss_f(x) /= input.width()*input.height();
    Buffer<float> loss_val = loss_f.realize(1);

    return 0;
}
