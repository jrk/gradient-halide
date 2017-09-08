// g++ bilateral_filter.cpp -g -O3 -I/usr/local/Cellar/libpng/1.6.32/include/ -I/usr/local/Cellar/jpeg/9b/include/ -I../tools -I ../include -L ../bin -lHalide -L/usr/local/Cellar/jpeg/9b/lib/ -ljpeg -L/usr/local/Cellar/libpng/1.6.32/lib/ -lpng -o bilateral_filter -std=c++11
// DYLD_LIBRARY_PATH=../bin ./bilateral_filter

#include "Halide.h"
#include "halide_image_io.h"

int main(int argc, char **argv) {
    Halide::Var x("x"), y("y"), c("c");
    Halide::Param<float> sigma_s("sigma_s", 5.0);
    Halide::Param<float> sigma_c("sigma_c", 50.0);
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("images/rgb.png");
    Halide::Func input_float;
    input_float(x, y, c) = Halide::cast<float>(input(x, y, c));
    Halide::Expr clamped_x = Halide::clamp(x, 0, input.width()-1);
    Halide::Expr clamped_y = Halide::clamp(y, 0, input.height()-1);
    Halide::Func clamped;
    clamped(x, y, c) = input_float(clamped_x, clamped_y, c);
    Halide::Expr output_numerator = 0.f;
    Halide::Expr output_denominator = 0.f;
    const int radius = 10;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            Halide::Expr squared_spatial_distance(dx * dx + dy * dy);
            Halide::Expr spatial_weight = exp(-squared_spatial_distance / (sigma_s * sigma_s));
            Halide::Expr color_distance = clamped(x + dx, y + dy, c) - clamped(x, y, c);
            Halide::Expr squared_color_distance = color_distance * color_distance;
            Halide::Expr color_weight = exp(-squared_color_distance / (sigma_c * sigma_c));
            Halide::Expr weight = color_weight * spatial_weight;
            output_numerator += weight * clamped(x + dx, y + dy, c);
            output_denominator += weight;
        }
    }
    Halide::Expr output = output_numerator / output_denominator;
    std::vector<Halide::Expr> derivative_exprs = Halide::derivative(output, {sigma_s, sigma_c});
    //std::cerr << "output:" << output << std::endl;
    //std::cerr << "exprs[0]:" << derivative_exprs[0] << std::endl;
    //std::cerr << "exprs[1]:" << derivative_exprs[1] << std::endl;

    Halide::Func output_float;
    output_float(x, y, c) = output;
    Halide::Func output_8bit;
    output_8bit(x, y, c) = Halide::cast<uint8_t>(output_float(x, y, c));
    Halide::Func d_sigma_s, d_sigma_c;
    d_sigma_s(x, y, c) = pow(derivative_exprs[0], 1/2.2f) * 8000.f;
    d_sigma_c(x, y, c) = pow(derivative_exprs[1], 1/2.2f) * 80000.f;
    Halide::Func d_sigma_s_8bit, d_sigma_c_8bit;
    d_sigma_s_8bit(x, y, c) = Halide::cast<uint8_t>(d_sigma_s(x, y, c));
    d_sigma_c_8bit(x, y, c) = Halide::cast<uint8_t>(d_sigma_c(x, y, c));

    Halide::Buffer<uint8_t> result = output_8bit.realize(input.width(), input.height(), 3);
    Halide::Tools::save_image(result, "filtered.png");
    Halide::Buffer<uint8_t> ds = d_sigma_s_8bit.realize(input.width(), input.height(), 3);
    Halide::Tools::save_image(ds, "ds.png");
    Halide::Buffer<uint8_t> dc = d_sigma_c_8bit.realize(input.width(), input.height(), 3);
    Halide::Tools::save_image(dc, "dc.png");
    return 0;
}
