#include <cstdio>
#include<random>

#include "hdrnet_slice.h"

#include "halide_benchmark.h"
#include "halide_image_io.h"
#include "HalideBuffer.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

void run_hdrnet() {
  const int grid_height = 16;
  const int grid_width = 16;
  const int grid_depth = 8;
  const int nparams = 3;  // last dim is the number of params
  Buffer<float> grid(
      grid_height, grid_width, grid_depth, nparams);

  std::random_device rd;
  std::mt19937 engine(rd());
  std::uniform_real_distribution<> dist(0, 1);

  for (int c = 0; c < nparams; ++c)
  for (int z = 0; z < grid_depth; ++z)
  for (int y = 0; y < grid_height; ++y)
  for (int x = 0; x < grid_width; ++x)
  {
    grid(x, y, z, c) = z*1.0/(grid_depth)*dist(engine);
  }

  Buffer<uint8_t> input = Halide::Tools::load_image("images/rgb.png");
  Buffer<float> guide(input.width(), input.height());

  // Make grayscale guide image
  float multiplier = 1.0f/255.0f;
  for (int c = 0; c < input.channels(); ++c)
  for (int y = 0; y < input.height(); ++y)
  for (int x = 0; x < input.width(); ++x)
  {
    guide(x, y) += multiplier*input(x, y, c);
  }

  Buffer<float> output(input.width(), input.height(), input.channels());
  hdrnet_slice(grid, guide, output);

  convert_and_save_image(output, "hdrnet_slice_output.png");
}

int main(int argc, char **argv) {
  run_hdrnet();
  return 0;
}
