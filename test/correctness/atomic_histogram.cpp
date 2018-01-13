#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    Func im, hist;
    Var x;
    RDom r(0, 1000);

    im(x) = cast<uint8_t>(x*x);

    hist(x) = 0;
    hist(im(r)) += 1;

    hist.compute_root();
    RVar ro, ri;
    hist.update().allow_race_conditions().split(r, ro, ri, 8).gpu_blocks(ro).gpu_threads(ri);

    Buffer<int> out = hist.realize(256);

    Buffer<int> correct(256);
    correct.fill(0);
    for (int i = 0; i < 1000; i++) {
        correct((i*i) & 255) += 1;
    }

    for (int i = 0; i < 256; i++) {
        if (out(i) != correct(i)) {
            printf("out(%d) = %d instead of %d\n", i, out(i), correct(i));
            return -1;
        }
    }

    return 0;
}
