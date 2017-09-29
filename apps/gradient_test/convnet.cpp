// g++ convnet.cpp -g -DDEBUG_RUNTIME -I/usr/local/Cellar/libpng/1.6.32/include/ -I/usr/local/Cellar/jpeg/9b/include/ -I../tools -I ../include -L ../lib -lHalide -L/usr/local/Cellar/jpeg/9b/lib/ -ljpeg -L/usr/local/Cellar/libpng/1.6.32/lib/ -lcurses -lz -lpng -o convnet -std=c++11
// DYLD_LIBRARY_PATH=../bin ./convnet

#include "Halide.h"
#include "halide_image_io.h"

int main(int argc, char **argv) {
    Halide::Var x("x"), y("y");
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("images/gray.png");
    Halide::Func input_float("input_float");
    input_float(x, y) = Halide::cast<float>(input(x, y));

    Halide::Expr clamped_x = Halide::clamp(x, 0, input.width()-1);
    Halide::Expr clamped_y = Halide::clamp(y, 0, input.height()-1);
    Halide::Func clamped("clamped");
    clamped(x, y) = input_float(clamped_x, clamped_y);

    Halide::Func conv_layer("conv_layer");

    float initial_weights[] = {1.f, 0.f, 0.f,
                               0.f, 1.f, 0.f,
                               0.f, 0.f, 1.f};
    Halide::Buffer<float> filter =
        Halide::Buffer<float>::make_interleaved(initial_weights, 3, 3, 1);
    Halide::Func filter_func("filter_func");
    filter_func(x, y) = filter(x, y);
    Halide::RDom r(0, 3, 0, 3);
    Halide::Func output("output");
    output(x, y) = 0.f;
    output(x, y) += clamped(x + r.x, y + r.y) * filter_func(r.x, r.y);
    Halide::RDom r_target(input);
    Halide::Expr diff = output(r_target.x, r_target.y) - input(r_target.x, r_target.y);
    Halide::Expr loss = diff * diff;

    std::map<std::string, Halide::Func> funcs = Halide::propagate_adjoints(loss);
    print_func(funcs["filter_func"]);
    // funcs[3].compile_to_lowered_stmt("df.html", {}, Halide::HTML);
    Halide::Buffer<float> df = funcs["filter_func"].realize(3, 3);
    std::cerr << "df(0, 0):" << df(0, 0) << std::endl;

    return 0;
}
