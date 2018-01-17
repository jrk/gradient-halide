#ifndef HALIDE_SIMPLE_AUTO_SCHEDULE_H
#define HALIDE_SIMPLE_AUTO_SCHEDULE_H

#include "Func.h"

#include <string>
#include <vector>
#include <map>
#include <set>

namespace Halide {

struct SimpleAutoscheduleOptions {
    bool gpu = false;
    int cpu_tile_width = 16;
    int cpu_tile_height = 16;
    int gpu_tile_width = 32;
    int gpu_tile_height = 16;
};

// Bounds are {min, max}
void simple_autoschedule(std::vector<Func> &outputs,
                         const std::map<std::string, int> &parameters,
                         const std::vector<std::vector<std::pair<int, int>>> &output_bounds,
                         const SimpleAutoscheduleOptions &options = SimpleAutoscheduleOptions(),
                         const std::set<std::string> &dont_inline = {},
                         const std::set<std::string> &skip_functions = {});
void simple_autoschedule(Func &output,
                         const std::map<std::string, int> &parameters,
                         const std::vector<std::pair<int, int>> &output_bounds,
                         const SimpleAutoscheduleOptions &options = SimpleAutoscheduleOptions(),
                         const std::set<std::string> &dont_inline = {},
                         const std::set<std::string> &skip_functions = {});

}

#endif 
