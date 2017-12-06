#include "Halide.h"
#include "halide_image_io.h"

#include <ctime>
#include <random>
#include <map>

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

void save_float_image(const Buffer<float> &img, const char *filename) {
    Buffer<uint8_t> buf(img.width(), img.height(), img.channels());
    for (int c = 0; c < buf.channels(); c++) {
        for (int y = 0; y < buf.height(); y++) {
            for (int x = 0; x < buf.width(); x++) {
                buf(x, y, c) = uint8_t(255.f * std::min(std::max(img(x, y, c), 0.f), 1.f));
            }
        }
    }
    save_image(buf, filename);
}

Buffer<float> copy(const Buffer<float> &buf) {
    Buffer<float> ret(buf.width(), buf.height(), buf.channels());
    for (int c = 0; c < buf.channels(); c++) {
        for (int y = 0; y < buf.height(); y++) {
            for (int x = 0; x < buf.width(); x++) {
                ret(x, y, c) = buf(x, y, c);
            }
        }
    }
    return ret;
}

void fill(const Buffer<float> &src, Buffer<float> &tgt) {
    for (int c = 0; c < tgt.channels(); c++) {
        for (int y = 0; y < tgt.height(); y++) {
            for (int x = 0; x < tgt.width(); x++) {
                tgt(x, y, c) = src(x, y, c);
            }
        }
    }
}

Var x("x"), y("y"), c("c");

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

Func cgInitializationFunc(const Buffer<float> &blurred,
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
                         regKernel(regKernel.width()  - regKernel.x - 1,
                                   regKernel.height() - regKernel.y - 1);
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
                      kernel(kernel.width()  - rKernel.x - 1, kernel.height() - rKernel.y - 1);
    Func ATAx0("A^TAx0");
    ATAx0(x, y, c) = 0.f;
    for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
        const Buffer<float> &regKernel = regKernels[i];
        RDom rRegKernel(regKernel);
        Func rKb("rKb");
        rKb(x, y, c) = 0.f;
        rKb(x, y, c) += clamped(x + rRegKernel.x - regKernel.width()  / 2,
                                y + rRegKernel.y - regKernel.height() / 2,
                                c) *
                         regKernel(regKernel.x, regKernel.y);
        Func rKTKb("rKTKb");
        rKTKb(x, y, c) = 0.f;
        rKTKb(x, y, c) += rKb(x + rRegKernel.x - regKernel.width()  / 2,
                              y + rRegKernel.y - regKernel.height() / 2,
                              c) *
                          regKernel(regKernel.width()  - regKernel.x - 1,
                                    regKernel.height() - regKernel.y - 1);
        ATAx0(x, y, c) += regKernelsWeight[i] * rKTKb(x, y, c);
    }

    Func r0("r0");
    r0(x, y, c) = ATb(x, y, c) - ATAx0(x, y, c);
    return r0;
}

std::tuple<Func, Func, Func> cgIterationFunc(const Buffer<float> &kernel,
                                             const Buffer<float> &xk,
                                             const Buffer<float> &r,
                                             const Buffer<float> &p,
                                             const std::vector<Func> &regKernelsWeight,
                                             const std::vector<Func> &regKernels,
                                             const RDom &rRegKernel,
                                             int regKernelWidth) {
    // A single iteration, outputs X, R, P
    // alpha = r^T * r / p^T A^T A p
    RDom rImage(xk);
    RDom rKernel(kernel);
    Func rTr("rTr");
    rTr()  = 0.f;
    rTr() += r(rImage.x, rImage.y, rImage.z) *
             r(rImage.x, rImage.y, rImage.z);
    Func Ap("Ap");
    Ap(x, y, c) = 0.f;
    Ap(x, y, c) += p(x + rKernel.x - kernel.width()  / 2,
                     y + rKernel.y - kernel.height() / 2,
                     c) *
                   kernel(rKernel.x, rKernel.y);
    Func ATAp("A^TAp");
    ATAp(x, y, c) = 0.f;
    ATAp(x, y, c) += Ap(x + rKernel.x - kernel.width()  / 2,
                        y + rKernel.y - kernel.height() / 2,
                        c) *
                  kernel(kernel.width() - rKernel.x - 1, kernel.height() - rKernel.y - 1);
    Func pTATAp("p^TA^TAp");
    pTATAp() = 0.f;
    pTATAp() += p(rImage.x, rImage.y, rImage.z) *
                ATAp(rImage.x, rImage.y, rImage.z);
    for (int regKernelId = 0; regKernelId < (int)regKernels.size(); regKernelId++) {
        Func rKp("rKp");
        rKp(x, y, c) = 0.f;
        rKp(x, y, c) += p(x + rRegKernel.x - kernel.width()  / 2,
                          y + rRegKernel.y - kernel.height() / 2,
                          c) *
                        regKernels[regKernelId](rRegKernel.x, rRegKernel.y);
        Func rKTrKp("rK^TrKp");
        rKTrKp(x, y, c) = 0.f;
        rKTrKp(x, y, c) += rKp(x + rRegKernel.x - kernel.width()  / 2,
                               y + rRegKernel.y - kernel.height() / 2,
                               c) *
                           regKernels[regKernelId](regKernelWidth - rRegKernel.x - 1,
                                                   regKernelWidth - rRegKernel.y - 1);
        pTATAp() += p(rImage.x, rImage.y, rImage.z) *
                    rKTrKp(rImage.x, rImage.y, rImage.z);
        pTATAp() *= (regKernelsWeight[regKernelId]() * regKernelsWeight[regKernelId]());
    }

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
    nRTnR()  = 0.f;
    nRTnR() += nextR(rImage.x, rImage.y, rImage.z) *
               nextR(rImage.x, rImage.y, rImage.z);
    Func beta("beta");
    beta() = nRTnR() / rTr();
    Func nextP("nextP");
    nextP(x, y, c) = nextR(x, y, c) + beta() * p(x, y, c);

    return std::make_tuple(nextX, nextR, nextP);
}

void train(int iteration,
           const Buffer<float> &image,
           const Buffer<float> &kernel,
           float noiseStddev,
           const std::vector<float> &regKernelsWeight,
           const std::vector<Buffer<float>> &regKernels,
           std::vector<float> &regKernelsWeightGrad,
           std::vector<Buffer<float>> &regKernelsGrad,
           std::mt19937 &rng) {
    Buffer<float> blurred = blur_image(image, kernel, noiseStddev, rng);
    char fnbuf[256];
    sprintf(fnbuf, "blurred_%d.png", iteration);
    save_float_image(blurred, fnbuf);

    RDom r_kernel(kernel);
    Func clamped = BoundaryConditions::repeat_edge(blurred);
    Func ATb("A^Tb");
    ATb(x, y, c)  = 0.f;
    ATb(x, y, c) += clamped(x + r_kernel.x - kernel.width()  / 2,
                            y + r_kernel.y - kernel.height() / 2,
                            c) *
                    kernel(kernel.width()  - r_kernel.x - 1,
                           kernel.height() - r_kernel.y - 1);
    Func Ax0("Ax0");
    Ax0(x, y, c)  = 0.f;
    Ax0(x, y, c) += clamped(x + r_kernel.x - kernel.width()  / 2,
                            y + r_kernel.y - kernel.height() / 2,
                            c) *
                 kernel(r_kernel.x, r_kernel.y);
    Func ATAx0("A^TAx0");
    ATAx0(x, y, c)  = 0.f;
    ATAx0(x, y, c) += Ax0(x + r_kernel.x - kernel.width()  / 2,
                          y + r_kernel.y - kernel.height() / 2,
                          c) *
                      kernel(kernel.width()  - r_kernel.x - 1, kernel.height() - r_kernel.y - 1);
    Func r0("r0");
    r0(x, y, c) = ATb(x, y, c) - ATAx0(x, y, c);
    r0.compute_root();
    ATb.compute_root();
    ATAx0.compute_root();
    Ax0.compute_root();
    Buffer<float> xk = copy(blurred);
    Buffer<float> r = r0.realize(blurred.width(), blurred.height(), blurred.channels());
    Buffer<float> p = copy(r);

    Func x_clamped = BoundaryConditions::repeat_edge(xk);
    Func r_clamped = BoundaryConditions::repeat_edge(r);
    Func p_clamped = BoundaryConditions::repeat_edge(p);

    std::vector<Func> regKernelsWeightFunc;
    std::vector<Func> regKernelsFunc;
    for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
        Func rKw("rKw");
        rKw() = regKernelsWeight[i];
        Func rK("rK");
        rK(x, y) = regKernels[i](x, y);
        regKernelsWeightFunc.push_back(rKw);
        regKernelsFunc.push_back(rK);
    }
    RDom rRegKernel(regKernels[0]);
    Func xkFunc = x_clamped;
    Func rFunc = r_clamped;
    Func pFunc = p_clamped;
    RDom rImage(xk);
    for (int i = 0; i < 1; i++) {
        std::tie(xkFunc, rFunc, pFunc) =
            cgIterationFunc(kernel, xkFunc, rFunc, pFunc,
                            regKernelsWeightFunc, regKernelsFunc,
                            rImage, rRegKernel, regKernels[0].width());
    }
    // estimate output sizes
    // Pipeline cgIteration({xkFunc});

    // Target compile_target = get_jit_target_from_environment();
    // cgIteration.auto_schedule(compile_target);

    // Realization realization =
    //     cgIteration.realize(blurred.width(), blurred.height(), blurred.channels());
    // fill(realization[0], xk);

    // sprintf(fnbuf, "deblurred_%d.png", iteration);
    // save_float_image(xk, fnbuf);

    Func target("target");
    target(x, y, c) = image(x, y, c);
    Func loss("loss");
    loss() = 0.f;
    Expr diff = target(rImage.x, rImage.y, rImage.z) -
                xkFunc(rImage.x, rImage.y, rImage.z);
    loss() += diff * diff;

    Derivative d = propagate_adjoints(loss);
    std::map<FuncKey, Func> adjoints = d.adjoints;
    std::vector<Func> pipelineFuncs;
    std::vector<Buffer<>> buffers;
    // apply_compute_root(xkFunc);
    xkFunc.estimate(x, 0, blurred.width())
          .estimate(y, 0, blurred.height())
          .estimate(c, 0, blurred.channels());
    pipelineFuncs.push_back(xkFunc);
    buffers.push_back(Buffer<float>(image.width(), image.height(), image.channels()));
    // for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
    //     pipelineFuncs.push_back(adjoints[FuncKey{regKernelsWeightFunc[i].name(), -1}]);
    //     // apply_compute_root(pipelineFuncs.back());
    //     pipelineFuncs.push_back(adjoints[FuncKey{regKernelsFunc[i].name(), -1}]);
    //     pipelineFuncs.back().estimate(x, 0, regKernels[i].width())
    //                         .estimate(y, 0, regKernels[i].height());
    //     // apply_compute_root(pipelineFuncs.back());
    //     buffers.push_back(Buffer<float>::make_scalar());
    //     buffers.push_back(Buffer<float>(regKernels[i].width(), regKernels[i].height()));
    // }
    Pipeline pipeline(pipelineFuncs);
    pipeline.auto_schedule(get_jit_target_from_environment());
    Realization realization(buffers);
    pipeline.realize(realization);

    Buffer<float> deblurred = buffers[0];
    sprintf(fnbuf, "deblurred_%d.png", iteration);
    save_float_image(deblurred, fnbuf);
    for (int i = 0; i < (int)regKernelsWeight.size(); i++) {
        Buffer<float> w = buffers[2 * i + 1].as<float>();
        regKernelsWeightGrad.push_back(w(0));
        regKernelsGrad.push_back(buffers[2 * i + 2].as<float>());
    }

    float error = 0.f;
    for (int c = 0; c < xk.channels(); c++) {
        for (int y = 0; y < xk.height(); y++) {
            for (int x = 0; x < xk.width(); x++) {
                error += (deblurred(x, y, c) - image(x, y, c)) *
                         (deblurred(x, y, c) - image(x, y, c));
            }
        }
    }
    error /= float(xk.width() * xk.height() * xk.channels());
    std::cout << "error:" << error << std::endl;
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
    float learningRate = 1e-4f;
    Buffer<float> kernel = generate_kernel(5, rng);
    std::vector<float> regKernelsWeight = {dxWeight, dyWeight};
    std::vector<Buffer<float>> regKernels = {dxKernel, dyKernel};
    Func r0 = cgInitializationFunc(blurred, kernel, regKernelsWeight, regKernels);

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
