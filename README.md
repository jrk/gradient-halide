This is an experimental branch of Halide (http://halide-lang.org) for making it
able to compute gradients. All the functionalities have been merged to the main repository (https://github.com/halide/Halide).

For a technical description see our paper, ["Differentiable Programming for
Image Processing and Deep Learning in Halide"](https://people.csail.mit.edu/tzumao/gradient_halide/).

For build instruction see the original repository (https://github.com/halide/Halide).

If you haven't used Halide before, it's probably a good idea to go through the [tutorials](http://halide-lang.org/tutorials/tutorial_introduction.html) first.

For examples on how to use the gradient extensions see the tests in
[test/correctness/autodiff.cpp](test/correctness/autodiff.cpp),
a simple polynomial function fitting example in [test/correctness/fit_function.cpp](test/correctness/fit_function.cpp),
and a more involved lens optimization application in [apps/derivatives/lens.cpp](apps/derivatives/lens.cpp).

For the implementation see [src/Derivative.h](src/Derivative.h),
[src/Derivative.cpp](src/Derivative.cpp),
[src/DerivativeUtils.h](src/DerivativeUtils.h),
[src/DerivativeUtils.cpp](src/DerivativeUtils.cpp).

We also implemented a new autoscheduler which takes a set of output Halide functions
and automatically schedules all the dependencies. See [src/SimpleAutoSchedule.h](src/SimpleAutoSchedule.h) and
the tests in [src/SimpleAutoSchedule.cpp](src/SimpleAutoSchedule.cpp).

A proper tutorial and the source code for applications demonstrated in the paper
will come soon.

If you have any questions, issues, bug report, feel free to open a Github issue
or email Tzu-Mao Li (tzumao@mit.edu).
