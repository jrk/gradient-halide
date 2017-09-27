// g++ multistage_filter.cpp -g -I/usr/local/Cellar/libpng/1.6.32/include/ -I/usr/local/Cellar/jpeg/9b/include/ -I../tools -I ../include -L ../lib -lHalide -L/usr/local/Cellar/jpeg/9b/lib/ -ljpeg -L/usr/local/Cellar/libpng/1.6.32/lib/ -lcurses -lz -lpng -o multistage_filter -std=c++11
// DYLD_LIBRARY_PATH=../bin ./multistage_filter

#include "Halide.h"
#include "halide_image_io.h"

int main(int argc, char **argv) {
    Halide::Var x("x"), y("y"), c("c");
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("images/rgb.png");
    Halide::Func input_float("input_float");
    input_float(x, y, c) = Halide::cast<float>(input(x, y, c));

    Halide::Expr clamped_x = Halide::clamp(x, 0, input.width()-1);
    Halide::Expr clamped_y = Halide::clamp(y, 0, input.height()-1);
    Halide::Func clamped("clamped");
    clamped(x, y, c) = input_float(clamped_x, clamped_y, c);

    Halide::Func blur_x("blur_x");
    blur_x(x, y, c) = (clamped(x-1, y, c) +
                       2 * clamped(x, y, c) +
                       clamped(x+1, y, c)) / 4;

    Halide::Func blur_y("blur_y");
    blur_y(x, y, c) = (blur_x(x, y-1, c) +
                       2 * blur_x(x, y, c) +
                       blur_x(x, y+1, c)) / 4;
    blur_y.compute_root();

    Halide::Buffer<uint8_t> target = Halide::Tools::load_image("images/rgb.png");
    Halide::Func diff_squared("diff_squared");
    Halide::Expr diff = blur_y(x, y, c) - target(x, y, c);
    diff_squared(x, y, c) = diff * diff;
    diff_squared.compute_root();

    diff_squared.compile_to_lowered_stmt("org_diff_squared.html", {}, Halide::HTML);

    std::vector<Halide::Func> funcs = Halide::propagate_adjoints(diff_squared);
    // funcs[0] = d_diff_squared = 1
    // funcs[1] = d_blur_y = 2 * (blur_y(x, y, c) - target(x, y, c))
    // funcs[2] = d_blur_x = (d_blur_y(x, y+1, c) +
    //                        2 * d_blur_y(x, y, c) +
    //                        d_blur_y(x, y-1, c)) / 4
    // funcs[3] = d_clamped = (d_blur_x(x+1, y, c) +
    //                         2 * d_blur_x(x, y, c) +
    //                         d_blur_x(x-1, y, c)) / 4
    // funcs[4] = d_input_float = d_clamped
    for (int i = 0; i < (int)funcs.size(); i++) {
        Halide::Func &func = funcs[i];
        std::cerr << "i:" << i << ", func.name():" << func.name() << std::endl;
        for (int update_id = -1; update_id < func.num_update_definitions(); update_id++) {
            if (update_id >= 0) {
                std::cerr << func.update_value(update_id) << std::endl;
            } else {
                std::cerr << func.value() << std::endl;
            }
        }
        //func.compute_root();
        func.compile_to_lowered_stmt(func.name() + ".html", {}, Halide::HTML);
        func.print_loop_nest();
    }

    return 0;
}
