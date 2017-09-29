// g++ convnet.cpp -g -DDEBUG_RUNTIME -I/usr/local/Cellar/libpng/1.6.32/include/ -I/usr/local/Cellar/jpeg/9b/include/ -I../tools -I ../include -L ../lib -lHalide -L/usr/local/Cellar/jpeg/9b/lib/ -ljpeg -L/usr/local/Cellar/libpng/1.6.32/lib/ -lcurses -lz -lpng -o convnet -std=c++11
// DYLD_LIBRARY_PATH=../bin ./convnet

#include "Halide.h"
#include "halide_image_io.h"

#include <random>

Halide::Buffer<float> initialize_filter_weights(const int kernel_width, const int kernel_height, std::mt19937 &rng) {
    const int kernel_center_x = kernel_width / 2;
    const int kernel_center_y = kernel_height / 2;
    Halide::Buffer<float> weights(kernel_width, kernel_height, "weights");
    for (int y = 0; y < kernel_height; y++) {
        for (int x = 0; x < kernel_width; x++) {
            float w = x == kernel_width / 2 && y == kernel_height / 2 ? 1.f : 0.f;
            //std::normal_distribution<float> dist(0.f, 0.1f);
            //w += dist(rng);
            weights(x, y) = w;
        }
    }
    return weights;
}

std::pair<Halide::Func, Halide::Func> conv_layer(const Halide::Func &input,
                                                 const Halide::Buffer<float> &filter) {
    Halide::Var x("x"), y("y");
    const int kernel_center_x = filter.width() / 2;
    const int kernel_center_y = filter.height() / 2;
    Halide::Func filter_func("filter_func");
    filter_func(x, y) = filter(x, y);
    Halide::RDom r(0, filter.width(), 0, filter.height());
    Halide::Func convolved("convolved");
    convolved(x, y) = 0.f;
    convolved(x, y) += input(x + r.x - kernel_center_x, y + r.y - kernel_center_y) * filter_func(r.x, r.y);
    return {convolved, filter_func};
}

Halide::Func relu_layer(const Halide::Func &input) {
    Halide::Func relu("relu");
    relu(input.args()) = max(input(input.args()), 0.f);
    return relu;
}

int main(int argc, char **argv) {
    Halide::Var x("x"), y("y");
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("images/gray.png");
    Halide::Func input_float("input_float");
    input_float(x, y) = Halide::cast<float>(input(x, y)) / 255.f;

    Halide::Expr clamped_x = Halide::clamp(x, 0, input.width()-1);
    Halide::Expr clamped_y = Halide::clamp(y, 0, input.height()-1);
    Halide::Func clamped("clamped");
    clamped(x, y) = input_float(clamped_x, clamped_y);

    std::mt19937 rng;
    Halide::Buffer<float> weights0 = initialize_filter_weights(5, 5, rng);
    Halide::Buffer<float> weights1 = initialize_filter_weights(5, 5, rng);
    Halide::Buffer<float> weights2 = initialize_filter_weights(5, 5, rng);

    Halide::Func convolved0, filter0;
    std::tie(convolved0, filter0) = conv_layer(clamped, weights0);
    Halide::Func relu0            = relu_layer(convolved0);
    Halide::Func convolved1, filter1;
    std::tie(convolved1, filter1) = conv_layer(relu0, weights1);
    Halide::Func relu1            = relu_layer(convolved1);

    Halide::RDom r_target(input);
    Halide::Expr diff = relu1(r_target.x, r_target.y) - clamped(r_target.x, r_target.y);
    Halide::Expr loss = diff * diff;

    Halide::Func loss_func;
    loss_func(x) = 0.f;
    loss_func(x) += loss;
    Halide::Buffer<float> loss_buf = loss_func.realize(1);
    std::cerr << "loss:" << loss_buf(0) << std::endl;

    std::map<std::string, Halide::Func> funcs = Halide::propagate_adjoints(loss);
    print_func(funcs[filter0.name()]);
    Halide::Buffer<float> df = funcs[filter0.name()].realize(5, 5);
    std::cerr << "df(0, 0):" << df(0, 0) << std::endl;

    return 0;
}
