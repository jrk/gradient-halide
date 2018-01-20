#include "SimpleAutoSchedule.h"
#include "DerivativeUtils.h"
#include "FindCalls.h"
#include "RealizationOrder.h"
#include "Simplify.h"
#include "Substitute.h"

namespace Halide {

using namespace Internal;

template <typename T>
std::vector<int> sort_indices(const std::vector<T> &v) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
         [&v](int i1, int i2) {return v[i1] < v[i2];});
    return idx;
}

void simple_autoschedule(std::vector<Func> &outputs,
                         const std::map<std::string, int> &parameters,
                         const std::vector<std::vector<std::pair<int, int>>> &output_bounds,
                         const SimpleAutoscheduleOptions &options,
                         const std::set<std::string> &dont_inline,
                         const std::set<std::string> &skip_functions) {
    user_assert(outputs.size() == output_bounds.size()) <<
        "[simple_autoschedule] outputs size and output_bounds size don't match \n";
    for (int i = 0; i < (int)output_bounds.size(); i++) {
        user_assert(outputs[i].dimensions() == (int)output_bounds[i].size()) <<
            "[simple_autoschedule] outputs dimensionality don't match with output_bounds. " <<
            outputs[i].name() << " " << outputs[i].dimensions() << " " << output_bounds[i].size() << "\n";
    }
    using namespace Internal;
    std::vector<FuncBounds> output_bounds_expr;
    for (const auto &bounds : output_bounds) {
        FuncBounds func_bounds;
        for (const auto &bound : bounds) {
            func_bounds.push_back(std::make_pair<Expr, Expr>(bound.first, bound.second));
        }
        output_bounds_expr.push_back(func_bounds);
    }
    std::map<std::string, Box> func_bounds = inference_bounds(outputs, output_bounds_expr);
    std::vector<Function> output_functions;
    output_functions.reserve(outputs.size());
    for (const auto &func : outputs) {
        output_functions.push_back(func.function());
    }
    std::map<std::string, Function> env;
    for (const auto &func : output_functions) {
        std::map<std::string, Function> local_env = find_transitive_calls(func);
        env.insert(local_env.begin(), local_env.end());
    }
    std::set<std::string> output_set;
    for (const auto &output : outputs) {
        output_set.insert(output.name());
    }
    std::vector<std::string> order = realization_order(output_functions, env).first;
    std::map<std::string, std::set<std::string>> dependencies;
    // Dependency analysis
    for (auto it = order.begin(); it != order.end(); it++) {
        Func func(env[*it]);
        std::set<std::string> calls = find_dependency(func);
        for (const auto &call : calls) {
            dependencies[call].insert(func.name());
        }
    }
    std::vector<std::string> new_order;
    for (auto it = order.begin(); it != order.end(); it++) {
        if (output_set.find(*it) != output_set.end() ||
                dont_inline.find(*it) != dont_inline.end()) {
            new_order.push_back(*it);
            continue;
        }
        if (skip_functions.find(*it) != skip_functions.end()) {
            continue;
        }
        Func callee(env[*it]);
        const std::set<std::string> &set = dependencies[*it];
        // If a function is called by only one function, and the callee doesn't have reductions
        // inline the callee
        if (set.size() == 1) {
            Func caller(env[*set.begin()]);
            bool has_rvars = false;
            for (int update_id = 0; update_id < callee.num_update_definitions(); update_id++) {
                if (callee.rvars(update_id).size() > 0) {
                    has_rvars = true;
                    break;
                }
            }
            if (!has_rvars) {
                continue;
            }
        }
        new_order.push_back(*it);
    }
    order = new_order;
    // Traverse from the consumers to the producers
    for (auto it = order.rbegin(); it != order.rend(); it++) {
        Func func(env[*it]);
        Box bounds = func_bounds[*it];
        std::vector<int> int_bounds;
        for (int i = 0; i < (int)bounds.size(); i++) {
            Interval interval = bounds[i];
            Expr extent = simplify(interval.max - interval.min + 1);
            for (const auto &param : parameters) {
                extent = substitute(param.first, Expr(param.second), extent);
            }
            extent = simplify(extent);
            const int64_t *extent_int = as_const_int(extent);
            user_assert(extent_int != nullptr) << "extent:" << extent << " is not constant.\n";
            int_bounds.push_back(*extent_int);
        }
        std::vector<int> bounds_rank = sort_indices(int_bounds);
        int dim_width = -1, dim_height = -1;
        if ((int)int_bounds.size() >= 2) {
            int last_index = bounds_rank.size() - 1;
            dim_width = std::min(bounds_rank[last_index], bounds_rank[last_index-1]);
            dim_height = std::max(bounds_rank[last_index], bounds_rank[last_index-1]);
        }

        func.compute_root();
        // initial definition is easy: everything is pure variables
        // just parallelize and vectorize if there are enough places to launch threads
        int tile_width = options.gpu ? options.gpu_tile_width : options.cpu_tile_width;
        int tile_height = options.gpu ? options.gpu_tile_height : options.cpu_tile_height;
        int tile_channel = options.gpu_tile_channel;
        int min_gpu_threads = 1;
        int min_cpu_threads = 8;
        int min_threads = options.gpu ? min_gpu_threads : min_cpu_threads;
        int vectorize_width = 8;
        bool tilable = false;
        if ((int)int_bounds.size() >= 2 &&
                int_bounds[dim_width] >= tile_width &&
                int_bounds[dim_height] >= tile_height &&
                (int_bounds[dim_width] / tile_width) *
                (int_bounds[dim_height] / tile_height) >= min_threads) {
            Var xo, yo, zo, xi, yi, zi;
            if (options.gpu) {
                bool first = true;
                Var fused_var;
                for (int i = 0; i < (int)func.args().size(); i++) {
                    if (i == dim_width || i == dim_height) {
                        continue;
                    }
                    if (first) {
                        fused_var = func.args()[i];
                        first = false;
                    } else {
                        func.fuse(fused_var, func.args()[i], fused_var);
                    }
                }
                if (first) {
                    // no fused_vars
                    func.reorder(func.args()[dim_width], func.args()[dim_height])
                        .gpu_tile(func.args()[dim_width], func.args()[dim_height],
                            xo, yo, xi, yi, tile_width, tile_height);
                } else {
                    func.reorder(func.args()[dim_width], func.args()[dim_height], fused_var)
                        .gpu_tile(func.args()[dim_width], func.args()[dim_height], fused_var,
                            xo, yo, zo, xi, yi, zi, tile_width, tile_height, tile_channel);
                }
            } else {
                Var tile_index;
                func.tile(func.args()[dim_width], func.args()[dim_height],
                        xo, yo, xi, yi, tile_width, tile_height)
                    .fuse(xo, yo, tile_index)
                    .parallel(tile_index)
                    .vectorize(xi, vectorize_width);
            }
            tilable = true;
        } else if (options.gpu) {
            // Even if there's not enough parallelism it's still a good idea to launch
            // gpu tiles to avoid memory copy
            if (func.args().size() == 0) {
                func.gpu_single_thread();
            } else {
                // Fuse variables
                std::vector<Var> fused_vars;
                fused_vars.push_back(func.args()[0]);
                int var_size = int_bounds[0];
                for (int i = 1; i < (int)func.args().size(); i++) {
                    Var new_var;
                    func.fuse(fused_vars.back(), func.args()[i], new_var);
                    fused_vars.push_back(new_var);
                    var_size *= int_bounds[i];
                }
                // Launch GPU threads
                Var block, thread;
                func.gpu_tile(fused_vars.back(), block, thread, std::min(var_size, 32));
            }
        }

        for (int update_id = 0; update_id < func.num_update_definitions(); update_id++) {
            std::vector<ReductionVariable> rvars =
                func.update(update_id).get_schedule().rvars();
            int rdim_width = -1;
            int rdim_height = -1;
            bool rvar_tilable = false;
            if (rvars.size() > 0) {
                std::vector<int> rvar_extents;
                Expr extent = rvars[0].extent;
                for (const auto &param : parameters) {
                    extent = substitute(param.first, Expr(param.second), extent);
                }
                extent = simplify(extent);
                const int64_t *extent_int = as_const_int(extent);
                user_assert(extent_int != nullptr) << "extent:" << extent << " is not constant.\n";
                rvar_extents.push_back(*extent_int);
                for (int arg_id = 1; arg_id < (int)rvars.size(); arg_id++) {
                    Expr extent = rvars[arg_id].extent;
                    for (const auto &param : parameters) {
                        extent = substitute(param.first, Expr(param.second), extent);
                    }
                    extent = simplify(extent);
                    const int64_t *extent_int = as_const_int(extent);
                    user_assert(extent_int != nullptr) << "extent:" << extent << " is not constant.\n";
                    rvar_extents.push_back(*extent_int);
                }
                std::vector<int> bounds_rank = sort_indices(rvar_extents);
                if ((int)int_bounds.size() >= 2) {
                    int last_index = bounds_rank.size() - 1;
                    int dwidth = std::min(bounds_rank[last_index], bounds_rank[last_index-1]);
                    int dheight = std::max(bounds_rank[last_index], bounds_rank[last_index-1]);
                    if (rvar_extents[dwidth] >= tile_width &&
                            rvar_extents[dheight] >= tile_height) {
                        rdim_width = dwidth;
                        rdim_height = dheight;
                    }
                }
            }
            // Unroll known, small rvars
            for (int rvar_id = 0; rvar_id < (int)rvars.size(); rvar_id++) {
                if (rvar_id != rdim_width && rvar_id != rdim_height) {
                    Expr extent = rvars[rvar_id].extent;
                    const int64_t *extent_int = as_const_int(extent);
                    if (extent_int != nullptr && *extent_int <= 8) {
                        func.update(update_id)
                            .unroll(RVar(rvars[rvar_id].var));
                    }
                }
            }
            rvar_tilable = rdim_width != -1 && rdim_height != -1;

            // If the domain of the image is small and the reduction is large, use rfactor
            // TODO: gracefully fallback if factorization is impossible
            if (!tilable && rvar_tilable) {
                if (rdim_width != -1 && rdim_height != -1) {
                    if (options.gpu) {
                        assert(rdim_width != rdim_height);
                        RVar rx(rvars[rdim_width].var);
                        RVar ry(rvars[rdim_height].var);
                        for (int level = 0; level < 1; level++) {
                            RVar rxo, rxi, ryo, ryi;
                            int size = 32;
                            func.update(update_id)
                                .split(rx, rxo, rxi, size)
                                .split(ry, ryo, ryi, size);
                            rx = rxo; ry = ryo;
                            Var xi, xo, yo;
                            Func interm = func.update(update_id)
                                              .rfactor({{rxi, xi},
                                                        {rxo, xo},
                                                        {ryo, yo}});                       
                            std::vector<VarOrRVar> new_order;
                            new_order.push_back(ryi);
                            for (const auto &arg : interm.update_args()) {
                                const Variable *var = arg.as<Variable>();
                                if (var != nullptr && !var->reduction_domain.defined() &&
                                        var->name != xi.name() &&
                                        var->name != xo.name() &&
                                        var->name != yo.name()) {
                                    new_order.push_back(Var(var->name));
                                }
                            }
                            new_order.push_back(xi);
                            new_order.push_back(xo);
                            new_order.push_back(yo);
                            Var txo, txi, tyo, tyi;
                            interm.compute_root()
                                  .reorder(xi, xo, yo)
                                  .gpu_blocks(xo, yo)
                                  .gpu_threads(xi);
                            interm.update()
                                  .reorder(new_order)
                                  .gpu_blocks(xo, yo)
                                  .gpu_threads(xi);
                        }
                    } else {
                        // Parallel on tiles and vectorize inside tile
                        RVar rxo, ryo, rxi, ryi;
                        func.update(update_id)
                            .split(RVar(rvars[rdim_width].var), rxo, rxi, tile_width)
                            .split(RVar(rvars[rdim_height].var), ryo, ryi, tile_height);
                        Var xo, yo, xi;
                        Func interm = func.update(update_id)
                                          .rfactor({{rxo, xo},
                                                    {ryo, yo},
                                                    {rxi, xi}});
                        Var tile_index;
                        std::vector<VarOrRVar> new_order;
                        new_order.push_back(ryi);
                        new_order.push_back(xi);
                        for (const auto &arg : interm.update_args()) {
                            const Variable *var = arg.as<Variable>();
                            if (var != nullptr && !var->reduction_domain.defined() &&
                                    var->name != xi.name() && var->name != xo.name() &&
                                    var->name != yo.name()) {
                                new_order.push_back(Var(var->name));
                            }
                        }
                        new_order.push_back(tile_index);
                        interm.compute_root()
                              .fuse(xo, yo, tile_index)
                              .parallel(tile_index)
                              .vectorize(xi);
                        interm.update()
                              .fuse(xo, yo, tile_index)
                              .reorder(new_order)
                              .parallel(tile_index)
                              .vectorize(xi);
                    }
                }
            }
            std::vector<Var> pure_args;
            std::vector<Expr> update_args = func.update_args(update_id);
            std::vector<int> pure_arg_bounds;
            for (int arg_id = 0; arg_id < (int)update_args.size(); arg_id++) {
                Expr arg = update_args[arg_id];
                const Variable *var = arg.as<Variable>();
                if (var != nullptr &&
                        !var->param.defined() &&
                        !var->image.defined() &&
                        !var->reduction_domain.defined()) {
                    pure_args.push_back(Var(var->name));
                    pure_arg_bounds.push_back(int_bounds[arg_id]);
                }
            }
            int pdim_width = -1;
            int pdim_height = -1;
            std::vector<int> bounds_rank = sort_indices(pure_arg_bounds);
            if ((int)bounds_rank.size() >= 2) {
                int last_index = bounds_rank.size() - 1;
                pdim_width = std::min(bounds_rank[last_index], bounds_rank[last_index-1]);
                pdim_height = std::max(bounds_rank[last_index], bounds_rank[last_index-1]);
            }

            if ((int)pure_arg_bounds.size() >= 2 &&
                     pure_arg_bounds[pdim_width] >= tile_width &&
                     pure_arg_bounds[pdim_height] >= tile_height &&
                    (pure_arg_bounds[pdim_width] / tile_width) *
                    (pure_arg_bounds[pdim_height] / tile_height) >= min_threads) {
                Var xo, yo, zo, xi, yi, zi;
                if (options.gpu) {
                    bool first = true;
                    Var fused_var;
                    for (int i = 0; i < (int)pure_args.size(); i++) {
                        if (i == pdim_width || i == pdim_height) {
                            continue;
                        }
                        if (first) {
                            fused_var = pure_args[i];
                            first = false;
                        } else {
                            func.update(update_id)
                                .fuse(fused_var, pure_args[i], fused_var);
                        }
                    }
                    if (first) {
                        // no fused_var
                        func.update(update_id)
                            .reorder(pure_args[pdim_width], pure_args[pdim_height])
                            .gpu_tile(pure_args[pdim_width], pure_args[pdim_height],
                                      xo, yo, xi, yi, tile_width, tile_height);

                    } else {
                        func.update(update_id)
                            .reorder(pure_args[pdim_width], pure_args[pdim_height], fused_var)
                            .gpu_tile(pure_args[pdim_width], pure_args[pdim_height], fused_var,
                                      xo, yo, zo, xi, yi, zi, tile_width, tile_height, tile_channel);
                    }
                } else {
                    Var tile_index;
                    func.update(update_id)
                        .tile(pure_args[pdim_width], pure_args[pdim_height],
                              xo, yo, xi, yi, tile_width, tile_height)
                        .fuse(xo, yo, tile_index)
                        .parallel(tile_index)
                        .vectorize(xi, vectorize_width);
                }
            } else if (!options.gpu && pure_args.size() > 0) {
                // On CPU, merge all pure variables and parallelize them
                Var fused_var = pure_args[0];
                for (int i = 1; i < (int)pure_args.size(); i++) {
                    func.update(update_id)
                        .fuse(fused_var, pure_args[i], fused_var);
                }
                func.update(update_id)
                    .parallel(fused_var);
            } else if (options.gpu) {
                // If the reduction domain is large enough, parallelize the reduction domain
                if (tilable && rvar_tilable) {
                    RVar xo, yo, xi, yi;
                    if (pure_args.size() > 0) {
                        Var zo, zi;
                        Var fused_var;
                        fused_var = pure_args[0];
                        for (int i = 1; i < (int)pure_args.size(); i++) {
                            func.update(update_id)
                                .fuse(fused_var, pure_args[i], fused_var);
                        }
                        func.update(update_id)
                            .allow_race_conditions()
                            .split(RVar(rvars[rdim_width].var), xo, xi, tile_width)
                            .split(RVar(rvars[rdim_height].var), yo, yi, tile_height)
                            .split(fused_var, zo, zi, tile_channel)
                            .reorder(xi, yi, zi, xo, yo, zo)
                            .gpu_blocks(xo, yo, zo)
                            .gpu_threads(xi, yi, zi);
                    } else {
                        func.update(update_id)
                            .allow_race_conditions()
                            .split(RVar(rvars[rdim_width].var), xo, xi, tile_width)
                            .split(RVar(rvars[rdim_height].var), yo, yi, tile_height)
                            .reorder(xi, yi, xo, yo)
                            .gpu_blocks(xo, yo)
                            .gpu_threads(xi, yi);
                    }
                } else {
                    // Even if there's not enough parallelism it's still a good idea to launch
                    // gpu tiles to avoid memory copy
                    if (pure_args.size() == 0) {
                        func.update(update_id).gpu_single_thread();
                    } else {
                        // Fuse variables
                        std::vector<Var> fused_vars;
                        fused_vars.push_back(pure_args[0]);
                        int var_size = pure_arg_bounds[0];
                        for (int i = 1; i < (int)pure_args.size(); i++) {
                            Var new_var;
                            func.update(update_id).fuse(fused_vars.back(), pure_args[i], new_var);
                            fused_vars.push_back(new_var);
                            var_size *= pure_arg_bounds[i];
                        }
                        // Launch GPU threads
                        Var block, thread;
                        func.update(update_id).gpu_tile(fused_vars.back(), block, thread, std::min(var_size, 32));
                    }
                }
            }

            // Special pattern: if we see f(r.x, r.y, ...) = f(r.x, r.y, ...) + ...
            // we will parallelize over r
            // only for CPU since we use atomics for gpu
            auto is_parallelizable_reduction = [&]() -> bool {
                if (update_args.size() == 0) {
                    return false;
                }
                for (const auto &arg : update_args) {
                    const Variable *var = arg.as<Variable>();
                    if (!(var != nullptr &&
                              !var->param.defined() &&
                              !var->image.defined() &&
                              var->reduction_domain.defined())) {
                        return false;
                    }
                }
                std::vector<Expr> update_vals = func.update_values(update_id).as_vector();
                for (const auto &val : update_vals) {
                    const Add *add = val.as<Add>();
                    if (add == nullptr) {
                        return false;
                    }
                    const Call *call = add->a.as<Call>();
                    if (call == nullptr) {
                        return false;
                    }
                    if (!call->func.defined()) {
                        return false;
                    }
                    Function called_func(call->func);
                    if (called_func.name() != func.name()) {
                        return false;
                    }
                    
                    for (int arg_id = 0; arg_id < (int)call->args.size(); arg_id++) {
                        const Variable *var = call->args[arg_id].as<Variable>();
                        if (!(var != nullptr &&
                                    !var->param.defined() &&
                                    !var->image.defined() &&
                                    var->reduction_domain.defined())) {
                            return false;
                        }
                        const Variable *update_var = update_args[arg_id].as<Variable>();
                        if (var->name != update_var->name) {
                            return false;
                        }
                    }
                }
                return true;
            };

            if (!options.gpu && is_parallelizable_reduction()) {
                std::vector<RVar> rvar_args;
                std::vector<int> rvar_arg_bounds;
                for (int arg_id = 0; arg_id < (int)update_args.size(); arg_id++) {
                    const Variable *var = update_args[arg_id].as<Variable>();
                    assert(var != nullptr);
                    rvar_args.push_back(RVar(var->name));
                    assert(var->reduction_domain.defined());
                    ReductionDomain rdom = var->reduction_domain;
                    const auto &domain = rdom.domain();
                    Expr extent = domain[arg_id].extent;
                    for (const auto &param : parameters) {
                        extent = substitute(param.first, Expr(param.second), extent);
                    }
                    extent = simplify(extent);
                    const int64_t *extent_int = as_const_int(extent);
                    user_assert(extent_int != nullptr) << "extent:" << extent << " is not constant.\n";
                    rvar_arg_bounds.push_back(*extent_int);
                }
                int rdim_width = -1;
                int rdim_height = -1;
                std::vector<int> bounds_rank = sort_indices(rvar_arg_bounds);
                if ((int)int_bounds.size() >= 2) {
                    int last_index = bounds_rank.size() - 1;
                    rdim_width = std::min(bounds_rank[last_index], bounds_rank[last_index-1]);
                    rdim_height = std::max(bounds_rank[last_index], bounds_rank[last_index-1]);
                }

                if ((int)rvar_arg_bounds.size() >= 2 &&
                         rvar_arg_bounds[rdim_width] >= tile_width &&
                         rvar_arg_bounds[rdim_height] >= tile_height &&
                        (rvar_arg_bounds[rdim_width] / tile_width) *
                        (rvar_arg_bounds[rdim_height] / tile_height) >= min_threads) {
                    RVar xo, yo, xi, yi;
                    RVar tile_index;
                    func.update(update_id)
                        .allow_race_conditions()
                        .tile(rvar_args[rdim_width], rvar_args[rdim_height],
                              xo, yo, xi, yi, tile_width, tile_height)
                        .fuse(xo, yo, tile_index)
                        .parallel(tile_index)
                        .vectorize(xi, vectorize_width);
                }
            }
        }
    }
}

void simple_autoschedule(Func &output,
                         const std::map<std::string, int> &parameters,
                         const std::vector<std::pair<int, int>> &output_bounds,
                         const SimpleAutoscheduleOptions &options,
                         const std::set<std::string> &dont_inline,
                         const std::set<std::string> &skip_functions) {
    std::vector<Func> outputs{output};
    std::vector<std::vector<std::pair<int, int>>> vector_output_bounds{output_bounds};
    return simple_autoschedule(outputs,
                               parameters,
                               vector_output_bounds,
                               options,
                               dont_inline,
                               skip_functions);
}

}
