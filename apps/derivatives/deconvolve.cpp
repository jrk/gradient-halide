#include "Halide.h"
#include "halide_image_io.h"

#include <ctime>
#include <random>
#include <map>
#include <chrono>

using namespace Halide;
using namespace Halide::Tools;
using Halide::BoundaryConditions::repeat_edge;

void apply_compute_root(Func F) {
    std::map<std::string, Halide::Internal::Function> flist =
        Halide::Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    for (auto fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        f.compute_root();
        // cout << "Warning: applying compute root schedule to " << f.name() << endl;
    }
    // cout << endl;
}

void save_float_image(const Buffer<float> &img, const char *filename, int index = -1) {
    Buffer<uint8_t> buf(img.width(), img.height(), img.channels());
    for (int c = 0; c < buf.channels(); c++) {
        for (int y = 0; y < buf.height(); y++) {
            for (int x = 0; x < buf.width(); x++) {
                if (index == -1) {
                    buf(x, y, c) = uint8_t(255.f * std::min(std::max(img(x, y, c), 0.f), 1.f));
                } else {
                    buf(x, y, c) = uint8_t(255.f * std::min(std::max(img(x, y, c, index), 0.f), 1.f));
                }
            }
        }
    }
    save_image(buf, filename);
}

Buffer<float> copy(const Buffer<float> &buf) {
    Buffer<float> ret(buf.width(), buf.height(), buf.channels(), buf.dim(3).extent());
    for (int n = 0; n < buf.dim(3).extent(); n++) {
        for (int c = 0; c < buf.channels(); c++) {
            for (int y = 0; y < buf.height(); y++) {
                for (int x = 0; x < buf.width(); x++) {
                    ret(x, y, c, n) = buf(x, y, c, n);
                }
            }
        }
    }
    return ret;
}

void fill(const Buffer<float> &src, Buffer<float> &tgt) {
    for (int n = 0; n < tgt.dim(3).extent(); n++) {
        for (int c = 0; c < tgt.channels(); c++) {
            for (int y = 0; y < tgt.height(); y++) {
                for (int x = 0; x < tgt.width(); x++) {
                    tgt(x, y, c, n) = src(x, y, c, n);
                }
            }
        }
    }
}

Var x("x"), y("y"), c("c"), n("n");
Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
Var rx("rx"), ry("ry");

Buffer<float> generate_kernel(int kernelWidth,
                              std::mt19937 &rng) {
    Buffer<float> kernel(kernelWidth, kernelWidth);
    float sum = 0.f;
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < kernel.width(); i++) {
        for (int j = 0; j < kernel.height(); j++) {
            kernel(i, j) = dist(rng);
            sum += kernel(i, j);
        }
    }
    for (int i = 0; i < kernel.width(); i++) {
        for (int j = 0; j < kernel.height(); j++) {
            kernel(i, j) /= sum;
        }
    }
    return kernel;
}

Buffer<float> blur_image(const Buffer<float> &image,
                         const Buffer<float> &kernel,
                         float noiseStddev,
                         std::mt19937 &rng) {
    Func clamped = BoundaryConditions::repeat_edge(image);
    RDom r_kernel(kernel);
    Func blurred("blurred");
    blurred(x, y, c)  = 0.f;
    blurred(x, y, c) += clamped(x + r_kernel.x - kernel.width()  / 2,
                                y + r_kernel.y - kernel.height() / 2,
                                c) *
                        kernel(r_kernel.x, r_kernel.y);
    blurred.compute_root();

    Buffer<float> result = blurred.realize(image.width(), image.height(), image.channels());
    std::normal_distribution<float> dist(0.f, noiseStddev);
    for (int c = 0; c < result.channels(); c++) {
        for (int y = 0; y < result.height(); y++) {
            for (int x = 0; x < result.width(); x++) {
                result(x, y, c) += dist(rng);
            }
        }
    }
    return result;
}

std::pair<Pipeline, Realization>
    cgInitializationFunc(const Buffer<float> &blurred,
                         const Buffer<float> &kernel,
                         const std::vector<float> &regKernelsWeight,
                         const std::vector<Buffer<float>> &regKernels) {
    // Initializing conjugate gradient
    // Want to solve A^TAx = A^Tb
    // A -> correlation with kernel
    // A^T -> convolution with kernel
    // Initializing r0 = A^Tb - A^TAx0
    RDom rKernel(kernel);
    Func clamped = BoundaryConditions::repeat_edge(blurred);
    Func KTb("K^Tb");
    KTb(x, y, c)  = 0.f;
    KTb(x, y, c) += clamped(x + rKernel.x - kernel.width()  / 2,
                            y + rKernel.y - kernel.height() / 2,
                            c) *
                    kernel(kernel.width()  - rKernel.x - 1,
                           kernel.height() - rKernel.y - 1);
    Func ATb("A^Tb");
    ATb(x, y, c) = 0.f;
    ATb(x, y, c) += KTb(x, y, c);
    for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
        const Buffer<float> &regKernel = regKernels[i];
        RDom rRegKernel(regKernel);
        Func rKTb("rK^Tb");
        rKTb(x, y, c) = 0.f;
        rKTb(x, y, c) += clamped(x + rRegKernel.x - regKernel.width()  / 2,
                                 y + rRegKernel.y - regKernel.height() / 2,
                                 c) *
                         regKernel(regKernel.width()  - rRegKernel.x - 1,
                                   regKernel.height() - rRegKernel.y - 1);
        ATb(x, y, c) += regKernelsWeight[i] * rKTb(x, y, c);
    }
    Func Kx0("Kx0");
    Kx0(x, y, c)  = 0.f;
    Kx0(x, y, c) += clamped(x + rKernel.x - kernel.width()  / 2,
                            y + rKernel.y - kernel.height() / 2,
                            c) *
                    kernel(rKernel.x, rKernel.y);
    Func KTKx0("K^TKx0");
    KTKx0(x, y, c)  = 0.f;
    KTKx0(x, y, c) += Kx0(x + rKernel.x - kernel.width()  / 2,
                          y + rKernel.y - kernel.height() / 2,
                          c) *
                      kernel(kernel.width()  - rKernel.x - 1,
                             kernel.height() - rKernel.y - 1);
    Func ATAx0("A^TAx0");
    ATAx0(x, y, c) = KTKx0(x, y, c);
    for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
        const Buffer<float> &regKernel = regKernels[i];
        RDom rRegKernel(regKernel);
        Func rKb("rKb");
        rKb(x, y, c) = 0.f;
        rKb(x, y, c) += clamped(x + rRegKernel.x - regKernel.width()  / 2,
                                y + rRegKernel.y - regKernel.height() / 2,
                                c) *
                         regKernel(rRegKernel.x, rRegKernel.y);
        Func rKTKb("rKTKb");
        rKTKb(x, y, c) = 0.f;
        rKTKb(x, y, c) += rKb(x + rRegKernel.x - regKernel.width()  / 2,
                              y + rRegKernel.y - regKernel.height() / 2,
                              c) *
                          regKernel(regKernel.width()  - rRegKernel.x - 1,
                                    regKernel.height() - rRegKernel.y - 1);
        ATAx0(x, y, c) += regKernelsWeight[i] * rKTKb(x, y, c);
    }

    Func x0("x0");
    x0(x, y, c) = blurred(x, y, c);
    Func r0("r0");
    r0(x, y, c) = ATb(x, y, c) - ATAx0(x, y, c);
    Func p0("p0");
    p0(x, y, c) = r0(x, y, c);
    Func xrp("xrp");
    xrp(x, y, c, n) = 0.f;
    xrp(x, y, c, 0) = x0(x, y, c);
    xrp(x, y, c, 1) = r0(x, y, c);
    xrp(x, y, c, 2) = p0(x, y, c);
    Pipeline pipeline({xrp});
    Buffer<> xrpBuf =
        Buffer<float>(blurred.width(), blurred.height(), blurred.channels(), 3);
    Realization realization({xrpBuf});
    xrp.estimate(x, 0, xrpBuf.width())
       .estimate(y, 0, xrpBuf.height())
       .estimate(c, 0, xrpBuf.channels())
       .estimate(n, 0, 3);
    pipeline.auto_schedule(get_jit_target_from_environment());
    return std::make_pair(pipeline, realization);
}

std::pair<Pipeline, Realization>
    cgIterationFuncFwd(const Buffer<float> &kernel,
                       const Buffer<float> &xrp,
                       const std::vector<float> &regKernelsWeight,
                       const std::vector<Buffer<float>> &regKernels) {
    // A single iteration, takes X, R, P and updates them
    std::vector<Func> regKernelsWeightFunc;
    std::vector<Func> regKernelsFunc;
    for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
        Func wFunc("wFunc");
        wFunc() = regKernelsWeight[i];
        Func rkFunc("rkFunc");
        rkFunc(x, y) = regKernels[i](x, y);
        regKernelsWeightFunc.push_back(wFunc);
        regKernelsFunc.push_back(rkFunc);
    }
    RDom rImage(0, xrp.width(), 0, xrp.height(), 0, xrp.channels());
    RDom rKernel(kernel);
    Func xrpBC = BoundaryConditions::repeat_edge(xrp);
    // Extract input
    Func xk("xk");
    xk(x, y, c) = xrpBC(x, y, c, 0);
    Func r("r");
    r(x, y, c) = xrpBC(x, y, c, 1);
    Func p("p");
    p(x, y, c) = xrpBC(x, y, c, 2);
    Func rTr("rTr");
    // alpha = r^T * r / p^T A^T A p
    rTr() = 0.f;
    rTr() += r(rImage.x, rImage.y, rImage.z) *
             r(rImage.x, rImage.y, rImage.z);
    // rTr() = print(rTr());
    Func Kp("Kp");
    Kp(x, y, c) = 0.f;
    Kp(x, y, c) += p(x + rKernel.x - kernel.width()  / 2,
                     y + rKernel.y - kernel.height() / 2,
                     c) *
                   kernel(rKernel.x, rKernel.y);
    Func KTKp("K^TKp");
    KTKp(x, y, c) = 0.f;
    KTKp(x, y, c) += Kp(x + rKernel.x - kernel.width()  / 2,
                        y + rKernel.y - kernel.height() / 2,
                        c) *
                     kernel(kernel.width()  - rKernel.x - 1,
                            kernel.height() - rKernel.y - 1);
    Func ATAp("A^TAp");
    ATAp(x, y, c) = KTKp(x, y, c);
    std::vector<Func> rKps;
    std::vector<Func> rKTrKps;
    for (int regKernelId = 0; regKernelId < (int)regKernels.size(); regKernelId++) {
        const Buffer<float> &regKernel = regKernels[regKernelId];
        RDom rRegKernel(regKernel);
        Func rK = regKernelsFunc[regKernelId];
        Func rKp("rKp");
        rKp(x, y, c) = 0.f;
        rKp(x, y, c) += p(x + rRegKernel.x - regKernel.width()  / 2,
                          y + rRegKernel.y - regKernel.height() / 2,
                          c) *
                        rK(rRegKernel.x, rRegKernel.y);
        Func rKTrKp("rK^TrKp");
        rKTrKp(x, y, c) = 0.f;
        rKTrKp(x, y, c) += rKp(x + rRegKernel.x - regKernel.width()  / 2,
                               y + rRegKernel.y - regKernel.height() / 2,
                               c) *
                           rK(regKernel.width()  - rRegKernel.x - 1,
                              regKernel.height() - rRegKernel.y - 1);
        rKps.push_back(rKp);
        rKTrKps.push_back(rKTrKp);
        Expr w = regKernelsWeightFunc[regKernelId]();
        ATAp(x, y, c) += rKTrKp(x, y, c) * w * w;
    }
    Func pTATAp("p^TA^TAp");
    pTATAp() = 0.f;
    pTATAp() += p(rImage.x, rImage.y, rImage.z) *
                ATAp(rImage.x, rImage.y, rImage.z);

    Func alpha("alpha");
    alpha() = rTr() / pTATAp();
    // x = x + alpha * p
    Func nextX("nextX");
    nextX(x, y, c) = xk(x, y, c) + alpha() * p(x, y, c);
    // r = r - alpha * A^TAp
    Func nextR("nextR");
    nextR(x, y, c) = r(x, y, c) - alpha() * ATAp(x, y, c);
    // beta = nextR^TnextR / r^Tr
    Func nRTnR("nRTnR");
    nRTnR() = 0.f;
    nRTnR() += nextR(rImage.x, rImage.y, rImage.z) *
               nextR(rImage.x, rImage.y, rImage.z);
    Func beta("beta");
    beta() = nRTnR() / rTr();
    Func nextP("nextP");
    nextP(x, y, c) = nextR(x, y, c) + beta() * p(x, y, c);
    Func nextXrp("nextXrp");
    nextXrp(x, y, c, n) = 0.f;
    nextXrp(x, y, c, 0) = nextX(x, y, c);
    nextXrp(x, y, c, 1) = nextR(x, y, c);
    nextXrp(x, y, c, 2) = nextP(x, y, c);

    Pipeline pipeline({nextXrp});
    Buffer<> xrpBuf =
        Buffer<float>(xrp.width(), xrp.height(), xrp.channels(), 3);
    Realization realization({xrpBuf});

    int tileX = 64;
    int tileY = 64;
    int vectorWidth = 8;
    nextXrp.compute_root()
           .tile(x, y, xo, yo, xi, yi, tileX, tileY)
           .parallel(yo)
           .vectorize(xi, vectorWidth);
    nextXrp.update(0)
           .tile(x, y, xo, yo, xi, yi, tileX, tileY)
           .parallel(yo)
           .vectorize(xi, vectorWidth);
    nextXrp.update(1)
           .tile(x, y, xo, yo, xi, yi, tileX, tileY)
           .parallel(yo)
           .vectorize(xi, vectorWidth);
    nextXrp.update(2)
           .tile(x, y, xo, yo, xi, yi, tileX, tileY)
           .parallel(yo)
           .vectorize(xi, vectorWidth);
    alpha.compute_root();
    Func pTATApInt = pTATAp.update().rfactor({{rImage.x, rx}, {rImage.y, ry}});
    pTATApInt.compute_root()
             .update()
             .parallel(ry, 8)
             .vectorize(rx, vectorWidth);
    ATAp.compute_root()
        .tile(x, y, xo, yo, xi, yi, tileX, tileY)
        .parallel(yo)
        .vectorize(xi, vectorWidth);
    for (int i = 0; i < (int)regKernels.size(); i++) {
        ATAp.update(i)
            .tile(x, y, xo, yo, xi, yi, tileX, tileY)
            .parallel(yo)
            .vectorize(xi, vectorWidth);
    }
    Kp.in()
      .compute_at(ATAp, xo)
      .vectorize(x, vectorWidth);
    for (int i = 0; i < (int)regKernels.size(); i++) {
        rKps[i].in()
               .compute_at(ATAp, xo)
               .vectorize(x, vectorWidth);
    }
    Func rTrInt = rTr.update().rfactor({{rImage.x, rx}, {rImage.y, ry}});
    rTrInt.compute_root()
          .update()
          .parallel(ry, 8)
          .vectorize(rx, vectorWidth);

    beta.compute_root();
    Func nRTnRInt = nRTnR.update().rfactor({{rImage.x, rx}, {rImage.y, ry}});
    nRTnRInt.compute_root()
            .update()
            .parallel(ry, 8)
            .vectorize(rx, vectorWidth);

    return std::make_pair(pipeline, realization);
}

std::pair<Pipeline, Realization>
    cgIterationFuncRev(const Buffer<float> &kernel,
                       const Buffer<float> &xrp,
                       const Buffer<float> &dXrp,
                       const std::vector<float> &regKernelsWeight,
                       const std::vector<Buffer<float>> &regKernels) {
    // A single iteration, takes X, R, P and updates them
    std::vector<Func> regKernelsWeightFunc;
    std::vector<Func> regKernelsFunc;
    for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
        Func wFunc("wFunc");
        wFunc() = regKernelsWeight[i];
        Func rkFunc("rkFunc");
        rkFunc(x, y) = regKernels[i](x, y);
        regKernelsWeightFunc.push_back(wFunc);
        regKernelsFunc.push_back(rkFunc);
    }
    RDom rImage(0, xrp.width(), 0, xrp.height(), 0, xrp.channels());
    RDom rKernel(kernel);
    Func xrpFunc("xrp");
    xrpFunc(x, y, c, n) = xrp(x, y, c, n);
    // Extract input
    Func xk("xk");
    xk(x, y, c) = xrpFunc(x, y, c, 0);
    Func r("r");
    r(x, y, c) = xrpFunc(x, y, c, 1);
    Func p("p");
    p(x, y, c) = xrpFunc(x, y, c, 2);
    Func clampedP = BoundaryConditions::repeat_edge(p,
        {std::make_pair(Expr(0), Expr(xrp.width())),
         std::make_pair(Expr(0), Expr(xrp.height())),
         std::make_pair(Expr(), Expr())});
    Func rTr("rTr");
    // alpha = r^T * r / p^T A^T A p
    rTr() = 0.f;
    rTr() += r(rImage.x, rImage.y, rImage.z) *
             r(rImage.x, rImage.y, rImage.z);
    // rTr() = print(rTr());
    Func Kp("Kp");
    Kp(x, y, c) = 0.f;
    Kp(x, y, c) += clampedP(x + rKernel.x - kernel.width()  / 2,
                            y + rKernel.y - kernel.height() / 2,
                            c) *
                   kernel(rKernel.x, rKernel.y);
    Func KTKp("K^TKp");
    KTKp(x, y, c) = 0.f;
    KTKp(x, y, c) += Kp(x + rKernel.x - kernel.width()  / 2,
                        y + rKernel.y - kernel.height() / 2,
                        c) *
                     kernel(kernel.width()  - rKernel.x - 1,
                            kernel.height() - rKernel.y - 1);
    Func ATAp("A^TAp");
    ATAp(x, y, c) = KTKp(x, y, c);
    std::vector<Func> rKps;
    std::vector<Func> rKTrKps;
    for (int regKernelId = 0; regKernelId < (int)regKernels.size(); regKernelId++) {
        const Buffer<float> &regKernel = regKernels[regKernelId];
        RDom rRegKernel(regKernel);
        Func rK = regKernelsFunc[regKernelId];
        Func rKp("rKp");
        rKp(x, y, c) = 0.f;
        rKp(x, y, c) += clampedP(x + rRegKernel.x - regKernel.width()  / 2,
                                 y + rRegKernel.y - regKernel.height() / 2,
                                 c) *
                        rK(rRegKernel.x, rRegKernel.y);
        Func rKTrKp("rK^TrKp");
        rKTrKp(x, y, c) = 0.f;
        rKTrKp(x, y, c) += rKp(x + rRegKernel.x - regKernel.width()  / 2,
                               y + rRegKernel.y - regKernel.height() / 2,
                               c) *
                           rK(regKernel.width()  - rRegKernel.x - 1,
                              regKernel.height() - rRegKernel.y - 1);
        rKps.push_back(rKp);
        rKTrKps.push_back(rKTrKp);
        Expr w = regKernelsWeightFunc[regKernelId]();
        ATAp(x, y, c) += rKTrKp(x, y, c) * w * w;

    }
    Func pTATAp("p^TA^TAp");
    pTATAp() = 0.f;
    pTATAp() += clampedP(rImage.x, rImage.y, rImage.z) *
                ATAp(rImage.x, rImage.y, rImage.z);

    Func alpha("alpha");
    alpha() = rTr() / pTATAp();
    // x = x + alpha * p
    Func nextX("nextX");
    nextX(x, y, c) = xk(x, y, c) + alpha() * clampedP(x, y, c);
    // r = r - alpha * A^TAp
    Func nextR("nextR");
    nextR(x, y, c) = r(x, y, c) - alpha() * ATAp(x, y, c);
    // beta = nextR^TnextR / r^Tr
    Func nRTnR("nRTnR");
    nRTnR() = 0.f;
    nRTnR() += nextR(rImage.x, rImage.y, rImage.z) *
               nextR(rImage.x, rImage.y, rImage.z);
    Func beta("beta");
    beta() = nRTnR() / rTr();
    Func nextP("nextP");
    nextP(x, y, c) = nextR(x, y, c) + beta() * clampedP(x, y, c);
    Func nextXrp("nextXrp");
    nextXrp(x, y, c, n) = 0.f;
    nextXrp(x, y, c, 0) = nextX(x, y, c);
    nextXrp(x, y, c, 1) = nextR(x, y, c);
    nextXrp(x, y, c, 2) = nextP(x, y, c);
    nextXrp.estimate(x, 0, xrp.width())
           .estimate(y, 0, xrp.height())
           .estimate(c, 0, xrp.channels())
           .estimate(n, 0, 3);

    Derivative d = propagate_adjoints(nextXrp, dXrp);
    Func nextDXrp = d.adjoints[FuncKey{xrpFunc.name(), -1}];
    // print_func(nextDXrp);
    nextDXrp.estimate(x, 0, xrp.width())
            .estimate(y, 0, xrp.height())
            .estimate(c, 0, xrp.channels())
            .estimate(n, 0, 3);
    std::vector<Func> pipelineFuncs;
    std::vector<Buffer<>> pipelineBuffers;
    pipelineFuncs.push_back(nextDXrp);
    pipelineBuffers.push_back(Buffer<float>(xrp.width(), xrp.height(), xrp.channels(), 3));
    for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
        Func dW = d.adjoints[FuncKey{regKernelsWeightFunc[i].name(), -1}];
        Func dRK = d.adjoints[FuncKey{regKernelsFunc[i].name(), -1}];
        dRK.estimate(x, 0, regKernels[i].width())
           .estimate(y, 0, regKernels[i].height());
        pipelineFuncs.push_back(dW);
        pipelineBuffers.push_back(Buffer<float>::make_scalar());
        pipelineFuncs.push_back(dRK);
        pipelineBuffers.push_back(Buffer<float>(regKernels[i].width(), regKernels[i].height()));
    }

    Pipeline pipeline(pipelineFuncs);
    Realization realization(pipelineBuffers);

    // int tileX = 64;
    // int tileY = 64;
    // int vectorWidth = 8;
    // nextXrp.compute_root()
    //        .tile(x, y, xo, yo, xi, yi, tileX, tileY)
    //        .parallel(yo)
    //        .vectorize(xi, vectorWidth);
    // nextXrp.update(0)
    //        .tile(x, y, xo, yo, xi, yi, tileX, tileY)
    //        .parallel(yo)
    //        .vectorize(xi, vectorWidth);
    // nextXrp.update(1)
    //        .tile(x, y, xo, yo, xi, yi, tileX, tileY)
    //        .parallel(yo)
    //        .vectorize(xi, vectorWidth);
    // nextXrp.update(2)
    //        .tile(x, y, xo, yo, xi, yi, tileX, tileY)
    //        .parallel(yo)
    //        .vectorize(xi, vectorWidth);
    // alpha.compute_root();
    // Func pTATApInt = pTATAp.update().rfactor({{rImage.x, rx}, {rImage.y, ry}});
    // pTATApInt.compute_root()
    //          .update()
    //          .parallel(ry, 8)
    //          .vectorize(rx, vectorWidth);
    // ATAp.compute_root()
    //     .tile(x, y, xo, yo, xi, yi, tileX, tileY)
    //     .parallel(yo)
    //     .vectorize(xi, vectorWidth);
    // for (int i = 0; i < (int)regKernels.size(); i++) {
    //     ATAp.update(i)
    //         .tile(x, y, xo, yo, xi, yi, tileX, tileY)
    //         .parallel(yo)
    //         .vectorize(xi, vectorWidth);
    // }
    // Kp.in()
    //   .compute_at(ATAp, xo)
    //   .vectorize(x, vectorWidth);
    // for (int i = 0; i < (int)regKernels.size(); i++) {
    //     rKps[i].in()
    //            .compute_at(ATAp, xo)
    //            .vectorize(x, vectorWidth);
    // }
    // Func rTrInt = rTr.update().rfactor({{rImage.x, rx}, {rImage.y, ry}});
    // rTrInt.compute_root()
    //       .update()
    //       .parallel(ry, 8)
    //       .vectorize(rx, vectorWidth);

    // beta.compute_root();
    // Func nRTnRInt = nRTnR.update().rfactor({{rImage.x, rx}, {rImage.y, ry}});
    // nRTnRInt.compute_root()
    //         .update()
    //         .parallel(ry, 8)
    //         .vectorize(rx, vectorWidth);

    pipeline.auto_schedule(get_jit_target_from_environment(true));
    return std::make_pair(pipeline, realization);
}

int main(int argc, char **argv) {
    Buffer<float> input = load_and_convert_image("images/rgb.png");

    int regKernelWidth = 3;
    Buffer<float> dxKernel(regKernelWidth, regKernelWidth);
    for (int y = 0; y < dxKernel.height(); y++) {
        for (int x = 0; x < dxKernel.width(); x++) {
            dxKernel(x, y) = 0.f;
        }
    }
    dxKernel(1, 1) = -1.f;
    dxKernel(2, 1) =  1.f;
    float dxWeight = 1.f;
    Buffer<float> dyKernel(regKernelWidth, regKernelWidth);
    for (int y = 0; y < dyKernel.height(); y++) {
        for (int x = 0; x < dyKernel.width(); x++) {
            dyKernel(x, y) = 0.f;
        }
    }
    dyKernel(1, 1) = -1.f;
    dyKernel(1, 2) =  1.f;
    float dyWeight = 1.f;

    std::mt19937 rng;
    // float learningRate = 1e-4f;
    Buffer<float> kernel = generate_kernel(5, rng);
    Buffer<float> blurred = blur_image(input, kernel, 0.0f, rng);
    std::vector<float> regKernelsWeight = {dxWeight, dyWeight};
    std::vector<Buffer<float>> regKernels = {dxKernel, dyKernel};
    auto initFunc
        = cgInitializationFunc(blurred, kernel, regKernelsWeight, regKernels);
    Pipeline initPipeline = std::get<0>(initFunc);
    Realization initRealization = std::get<1>(initFunc);
    initPipeline.realize(initRealization);
    Buffer<float> xrp = initRealization[0];
    auto fwdIterFunc
        = cgIterationFuncFwd(kernel, xrp, regKernelsWeight, regKernels);
    Pipeline fwdIterPipeline = std::get<0>(fwdIterFunc);
    Realization fwdIterRealization = std::get<1>(fwdIterFunc);
    fwdIterPipeline.compile_jit(get_jit_target_from_environment());
    save_float_image(xrp, "x.png", 0);
    // Fwd pass
    std::vector<Buffer<float>> xrps;
    auto fwdStart = std::chrono::system_clock::now();
    for (int cgIter = 0; cgIter < 10; cgIter++) {
        xrps.push_back(copy(xrp));
        fwdIterPipeline.realize(fwdIterRealization);
        Buffer<float> nextXRP = fwdIterRealization[0];
        fill(nextXRP, xrp);
        // char buf[256];
        // sprintf(buf, "x_%d.png", cgIter);
        // save_float_image(xrp, buf, 0);
    }
    auto fwdEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> fwdSlapsedSeconds = fwdEnd - fwdStart;
    std::cout << "Forward time:" << fwdSlapsedSeconds.count() << std::endl;
    Buffer<float> dXrp(xrp.width(), xrp.height(), xrp.channels(), 3);
    for (int c = 0; c < xrp.channels(); c++) {
        for (int y = 0; y < xrp.height(); y++) {
            for (int x = 0; x < xrp.width(); x++) {
                dXrp(x, y, c, 0) = 2.f * (xrp(x, y, c, 0) - input(x, y, c));
            }
        }
    }
    for (int n = 1; n < 3; n++) {
        for (int c = 0; c < xrp.channels(); c++) {
            for (int y = 0; y < xrp.height(); y++) {
                for (int x = 0; x < xrp.width(); x++) {
                    dXrp(x, y, c, n) = 0.f;
                }
            }
        }
    }

    std::vector<Buffer<float>> dW;
    std::vector<Buffer<float>> dRK;
    for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
        dW.push_back(Buffer<float>::make_scalar());
        dRK.push_back(Buffer<float>(regKernels[i].width(), regKernels[i].height()));
        dW[i](0) = 0.f;
        for (int y = 0; y < regKernels[i].height(); y++) {
            for (int x = 0; x < regKernels[i].width(); x++) {
                dRK[i](x, y) = 0.f;
            }
        }
    }
    auto revIterFunc
        = cgIterationFuncRev(kernel, xrp, dXrp, regKernelsWeight, regKernels);
    Pipeline revIterPipeline = std::get<0>(revIterFunc);
    Realization revIterRealization = std::get<1>(revIterFunc);
    revIterPipeline.compile_jit(get_jit_target_from_environment());
    // Rev pass
    auto revStart = std::chrono::system_clock::now();
    for (int cgIter = 9; cgIter >= 0; cgIter--) {
        fill(xrps[cgIter], xrp);
        revIterPipeline.realize(revIterRealization);
        Buffer<float> nextDXrp = revIterRealization[0];
        fill(nextDXrp, dXrp);
        for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
            Buffer<float> dWi = revIterRealization[2 * i + 1];
            Buffer<float> dRKi = revIterRealization[2 * i + 2];
            dW[i](0) += dWi(0);
            for (int y = 0; y < regKernels[i].height(); y++) {
                for (int x = 0; x < regKernels[i].width(); x++) {
                    dRK[i](x, y) += dRKi(x, y);
                }
            }
        }
    }
    auto revEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> revSlapsedSeconds = revEnd - revStart;
    std::cout << "Reverse time:" << revSlapsedSeconds.count() << std::endl;

    // std::cerr << "dxWeight:" << dW[0](0) << std::endl;
    // std::cerr << "dyWeight:" << dW[1](0) << std::endl;

    // for (int iter = 0; iter < 1; iter++) {
    //     std::vector<float> regKernelsWeightGrad;
    //     std::vector<Buffer<float>> regKernelsGrad;
    //     train(iter, input, kernel, 0.01f,
    //           {dxWeight, dyWeight},
    //           {dxKernel, dyKernel},
    //           regKernelsWeightGrad,
    //           regKernelsGrad,
    //           rng);
    //     dxWeight -= learningRate * regKernelsWeightGrad[0];
    //     dyWeight -= learningRate * regKernelsWeightGrad[1];
    //     for (int y = 0; y < dyKernel.height(); y++) {
    //         for (int x = 0; x < dyKernel.width(); x++) {
    //             dxKernel(x, y) -= learningRate * regKernelsGrad[0](x, y);
    //             dyKernel(x, y) -= learningRate * regKernelsGrad[1](x, y);
    //         }
    //     }
    //     std::cout << "dxWeight:" << dxWeight << std::endl;
    //     std::cout << "dyWeight:" << dyWeight << std::endl;
    //     for (int y = 0; y < dyKernel.height(); y++) {
    //         for (int x = 0; x < dyKernel.width(); x++) {
    //             std::cout << dxKernel(x, y) << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     for (int y = 0; y < dyKernel.height(); y++) {
    //         for (int x = 0; x < dyKernel.width(); x++) {
    //             std::cout << dyKernel(x, y) << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    return 0;
}
