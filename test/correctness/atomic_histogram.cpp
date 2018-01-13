#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    Func im, hist;
    Var x;
    RDom r(0, 1000);

    im(x) = cast<uint8_t>(x*x);

    hist(x) = 0.0f;
    hist(im(r)) += 1.0f;

    hist.compute_root();
    RVar ro, ri;
    hist.update().allow_race_conditions().split(r, ro, ri, 8).gpu_blocks(ro).gpu_threads(ri);

    Buffer<float> out = hist.realize(256);

    Buffer<float> correct(256);
    correct.fill(0.0f);
    for (int i = 0; i < 1000; i++) {
        correct((i*i) & 255) += 1.0f;
    }

    for (int i = 0; i < 256; i++) {
        if (out(i) != correct(i)) {
            printf("out(%d) = %f instead of %f\n", i, out(i), correct(i));
            return -1;
        }
    }

    return 0;
}
