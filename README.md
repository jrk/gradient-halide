This is an experimental branch of Halide (http://halide-lang.org) for making it
able to compute gradients.

For a technical description see our paper, ["Differentiable Programming for
Image Processing and Deep Learning in Halide"](https://people.csail.mit.edu/tzumao/gradient_halide/).

For build instruction see the original repository (https://github.com/halide/Halide).

For examples on how to use the extensions see the tests in
[test/correctness/autodiff.cpp],
a simple polynomial function fitting example in [test/correctness/fit_function.cpp],
and a more involved lens optimization application in [apps/derivatives/lens.cpp].

A proper tutorial and the source code for applications demonstrated in the paper
will come soon.

If you have any questions, issues, bug report, feel free to open an Github issue
or email Tzu-Mao Li (tzumao@mit.edu).