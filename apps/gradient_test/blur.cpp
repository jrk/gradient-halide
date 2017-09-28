// g++ convnet.cpp -g -DDEBUG_RUNTIME -I/usr/local/Cellar/libpng/1.6.32/include/ -I/usr/local/Cellar/jpeg/9b/include/ -I../tools -I ../include -L ../lib -lHalide -L/usr/local/Cellar/jpeg/9b/lib/ -ljpeg -L/usr/local/Cellar/libpng/1.6.32/lib/ -lcurses -lz -lpng -o convnet -std=c++11
// DYLD_LIBRARY_PATH=../bin ./convnet

#include "Halide.h"
#include "halide_image_io.h"

using Halide::Var;
using Halide::Expr;
using Halide::Buffer;
using Halide::Func;
using Halide::BoundaryConditions::repeat_edge;

int main(int argc, char **argv) {
    Var x("x"), y("y");
    Buffer<uint8_t> input = Halide::Tools::load_image("images/gray.png");
    std::cout << "Loaded image with size " << input.width() << "x" << input.height() << std::endl;

    // Func clamped("clamped");
    // clamped(x,y) = repeat_edge(input)(x, y);

    Func input_float("input_float");
    input_float(x, y) = Halide::cast<float>(input(x, y))/255.0f;


    Func blur_x("blur_x");
    blur_x(x, y) = (input_float(x, y) + input_float(x+1, y) + input_float(x+2, y))/3.0f;

    Func blur_y("blur_y");
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+1))/3.0f;

    // // Simple updates
    // blur_y(0, 0) += 1.0f;
    // blur_y(0, 1) += 2.0f;
    // blur_y(1, 1) += 3.0f;

    Halide::RDom r(0, input.width()-2, 0, input.height()-2);  // No proper boundary conditions yet
    Expr diff = blur_y(r.x, r.y) - input_float(r.x, r.y);
    Expr loss = diff * diff;

    Func loss_f("loss_f");
    loss_f(x) = 0.0f;
    loss_f(x) += loss;
    // loss_f(x) /= input.width()*input.height();
    Buffer<float> loss_val = loss_f.realize(1);

    std::cout << "loss = " << loss_val(0) << std::endl;

    std::vector<Func> funcs = Halide::propagate_adjoints(loss);

    // funcs[3] = d output / d filter_func
    // print_func(funcs[3]);

    // // funcs[3].compile_to_lowered_stmt("df.html", {}, Halide::HTML);
    // Halide::Buffer<float> df = funcs[funcs.size() - 1].realize(3, 3);
    // std::cerr << "df(0, 0):" << df(0, 0) << std::endl;

    return 0;
}
