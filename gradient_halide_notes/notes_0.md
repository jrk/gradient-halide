```C++
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
    //blur_x.compute_root();

    Halide::Func blur_y("blur_y");
    blur_y(x, y, c) = (blur_x(x, y-1, c) +
                       2 * blur_x(x, y, c) +
                       blur_x(x, y+1, c)) / 4;
    //blur_y.compute_root();

    Halide::Buffer<uint8_t> target = Halide::Tools::load_image("images/rgb.png");
    Halide::Func diff_squared("diff_squared");
    Halide::Expr diff = blur_y(x, y, c) - target(x, y, c);
    diff_squared(x, y, c) = diff * diff;
    //diff_squared.compute_root();

    std::vector<Halide::Func> funcs = Halide::propagate_adjoints(diff_squared);
    // funcs[0] = diff_squared_d = 1
    // funcs[1] = blur_y_d = 2 * (blur_y(x, y, c) - target(x, y, c))
    // funcs[2] = blur_x_d = (blur_y_d(x, y+1, c) +
    //                        2 * blur_y_d(x, y, c) +
    //                        blur_y_d(x, y-1, c)) / 4
    // funcs[3] = clamped_d = (blur_x_d(x+1, y, c) +
    //                         2 * blur_x_d(x, y, c) +
    //                         blur_x_d(x-1, y, c)) / 4
    // funcs[4] = input_float_d = clamped_d
```

No compute_root: everything is inlined.

[funcs[1].compile_to_lowered_stmt()](blur_y_d_all_inline.html)

```
funcs[1].print_loop_nest():

produce blur_x_d___1:
  for c:
    for y:
      for x:
        blur_x_d___1(...) = ...
```

If we uncomment ``` blur_y.compute_root() ```

[funcs[1].compile_to_lowered_stmt()](blur_y_d_blur_y_computed.html)

```
produce blur_y:
  for c:
    for y:
      for x:
        blur_y(...) = ...
consume blur_y:
  produce blur_x_d___1:
    for c:
      for y:
        for x:
          blur_x_d___1(...) = ...
```

If we also ``` funcs[2].compute_root() ``` (precompute blur_x_d)

[funcs[3].compile_to_lowered_stmt()](clamped_d_computed.html)

```
produce blur_y:
  for c:
    for y:
      for x:
        blur_y(...) = ...
consume blur_y:
  produce blur_x_d___1:
    for c:
      for y:
        for x:
          blur_x_d___1(...) = ...
  consume blur_x_d___1:
    produce clamped_d___1:
      for c:
        for y:
          for x:
            clamped_d___1(...) = ...
```