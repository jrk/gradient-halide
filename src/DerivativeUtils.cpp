#include "DerivativeUtils.h"

#include "IRVisitor.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IREquality.h"
#include "Simplify.h"

namespace Halide {
namespace Internal {

class VariableFinder : public IRGraphVisitor {
public:
    using IRGraphVisitor::visit;
    bool find(const Expr &expr, const std::string &_var_name) {
        visited.clear();
        var_name = _var_name;
        found = false;
        expr.accept(this);
        return found;
    }

    void visit(const Variable *op) {
        if (op->name == var_name) {
            found = true;
        }
    }

private:
    std::string var_name;
    bool found;
};

bool has_variable(const Expr &expr, const std::string &name) {
    VariableFinder finder;
    return finder.find(expr, name);
}



class LetFinder : public IRGraphVisitor {
public:
    using IRGraphVisitor::visit;
    bool find(const Expr &expr, const std::string &_var_name) {
        visited.clear();
        var_name = _var_name;
        found = false;
        expr.accept(this);
        return found;
    }

    void visit(const Let *op) {
        if (op->name == var_name) {
            found = true;
        }
        op->value->accept(this);
        op->body->accept(this);
    }

private:
    std::string var_name;
    bool found;
};

bool has_let_defined(const Expr &expr, const std::string &name) {
    LetFinder finder;
    return finder.find(expr, name);
}

class LetRemover : public IRMutator {
public:
    Expr remove(const Expr &expr) {
        return mutate(expr);
    }

    void visit(const Let *op) {
        expr = mutate(op->body);
    }
};

Expr remove_let_definitions(const Expr &expr) {
    LetRemover remover;
    return remover.remove(expr);
}

class VariableGatherer : public IRGraphVisitor {
public:
    using IRGraphVisitor::visit;
    std::vector<std::string> gather(const Expr &expr,
                                    const std::vector<std::string> &_filter) {
        filter = _filter;
        visited.clear();
        variables.clear();
        expr.accept(this);
        return variables;
    }

    void visit(const Variable *op) {
        for (const auto &pv : filter) {
            if (op->name == pv) {
                variables.push_back(op->name);    
            }
        }
    }

private:
    std::vector<std::string> variables;
    std::vector<std::string> filter;
};

std::vector<std::string> gather_variables(const Expr &expr,
		const std::vector<std::string> &filter) {
	VariableGatherer gatherer;
	return gatherer.gather(expr, filter);
}

class RVarGatherer : public IRGraphVisitor {
public:
    using IRGraphVisitor::visit;
    std::map<std::string, std::pair<Expr, Expr>> gather(const Expr &expr) {
        visited.clear();
        rvar_map.clear();
        expr.accept(this);
        return rvar_map;
    }

    void visit(const Variable *op) {
        if (op->reduction_domain.defined()) {
            for (const ReductionVariable &rv : op->reduction_domain.domain()) {
                if (rv.var == op->name) {
                    rvar_map[op->name] = std::make_pair(rv.min, rv.extent);
                    return;
                }
            }
            internal_error << "Unknown reduction variable encountered";
        }
    }
private:
    std::map<std::string, std::pair<Expr, Expr>> rvar_map;
};

std::map<std::string, std::pair<Expr, Expr>> gather_rvariables(const Expr &expr) {
	RVarGatherer gatherer;
	return gatherer.gather(expr);
}


std::pair<Expr, Expr> get_min_max_bounds(const Expr &expr,
                                         const std::vector<Var> &current_args,
                                         const RDom &current_bounds,
                                         const int index,
                                         const Scope<Expr> &scope) {
    if (expr.get()->node_type == IRNodeType::Add) {
        const Add *op = expr.as<Add>();
        const std::pair<Expr, Expr> a_bounds =
            get_min_max_bounds(op->a, current_args, current_bounds, index, scope);
        const std::pair<Expr, Expr> b_bounds =
            get_min_max_bounds(op->b, current_args, current_bounds, index, scope);
        //debug(0) << "  " << index << " bounds for Add\n";
        return {a_bounds.first + b_bounds.first, a_bounds.second + b_bounds.second};
    } else if (expr.get()->node_type == IRNodeType::Sub) {
        const Sub *op = expr.as<Sub>();
        const std::pair<Expr, Expr> a_bounds =
            get_min_max_bounds(op->a, current_args, current_bounds, index, scope);
        const std::pair<Expr, Expr> b_bounds =
            get_min_max_bounds(op->b, current_args, current_bounds, index, scope);
        //debug(0) << "  " << index << " bounds for Sub\n";
        return {a_bounds.first - b_bounds.second, a_bounds.second - b_bounds.first};
    } else if (expr.get()->node_type == IRNodeType::Mul) {
        const Mul *op = expr.as<Mul>();
        const std::pair<Expr, Expr> a_bounds =
            get_min_max_bounds(op->a, current_args, current_bounds, index, scope);
        const std::pair<Expr, Expr> b_bounds =
            get_min_max_bounds(op->b, current_args, current_bounds, index, scope);
        //debug(0) << "  " << index << " bounds for Sub\n";
        return {a_bounds.first * b_bounds.first, a_bounds.second * b_bounds.second};
    } else if (expr.get()->node_type == IRNodeType::Variable) {
        const Variable *var = expr.as<Variable>();
        if (var->reduction_domain.defined()) {
            ReductionVariable rvar = var->reduction_domain.domain()[index];
            //debug(0) << "  " << index << " bounds for Rvar\n";
            return {rvar.min, rvar.min + rvar.extent - 1};
        } else {
            //debug(0) << "  " << index << " bounds for Var\n";
            for (int i = 0; i < (int)current_args.size(); i++) {
                if (current_args[i].name() == var->name) {
                    return {current_bounds[i].min(), current_bounds[i].extent()};
                }
            }
            if (scope.contains(var->name)) {
                return get_min_max_bounds(scope.get(var->name),
                                          current_args,
                                          current_bounds,
                                          index,
                                          scope);
            }
        }
        // Treat it as a constant (XXX: is this correct?)
        return {expr, expr};
    } else if (expr.get()->node_type == IRNodeType::Max) {
        const Max *op = expr.as<Max>();
        const std::pair<Expr, Expr> a_bounds =
            get_min_max_bounds(op->a, current_args, current_bounds, index, scope);
        const std::pair<Expr, Expr> b_bounds =
            get_min_max_bounds(op->b, current_args, current_bounds, index, scope);
        //debug(0) << "  " << index << " bounds for Max\n";
        return {max(a_bounds.first, b_bounds.first), max(a_bounds.second, b_bounds.second)};
    } else if (expr.get()->node_type == IRNodeType::Min) {
        const Min *op = expr.as<Min>();
        const std::pair<Expr, Expr> a_bounds =
            get_min_max_bounds(op->a, current_args, current_bounds, index, scope);
        const std::pair<Expr, Expr> b_bounds =
            get_min_max_bounds(op->b, current_args, current_bounds, index, scope);
        //debug(0) << "  " << index << " bounds for Min\n";
        return {min(a_bounds.first, b_bounds.first), min(a_bounds.second, b_bounds.second)};
    } else if (expr.get()->node_type == IRNodeType::IntImm) {
        //debug(0) << "  " << index << " bounds for IntImm\n";
        return {expr, expr};
    } else if (expr.get()->node_type == IRNodeType::FloatImm) {
        return {expr, expr};
    } else if (expr.get()->node_type == IRNodeType::Cast) {
        const Cast *op = expr.as<Cast>();
        const std::pair<Expr, Expr> bounds =
            get_min_max_bounds(op->value, current_args, current_bounds, index, scope);
        return {cast(op->type, bounds.first), cast(op->type, bounds.second)};
    } else if (expr.get()->node_type == IRNodeType::Call) {
        const Call *op = expr.as<Call>();
        if (op->call_type == Call::PureExtern) {
            if (op->name == "floor_f32") {
                internal_assert(op->args.size() == 1);
                const std::pair<Expr, Expr> bounds =
                    get_min_max_bounds(op->args[0], current_args, current_bounds, index, scope);
                return {floor(bounds.first), floor(bounds.second)};
            }
        } else if (op->call_type == Call::Halide) {
            return {std::numeric_limits<float>::min(),
                    std::numeric_limits<float>::max()};
        }
    } else if (expr.get()->node_type == IRNodeType::Let) {
        //const Let *op = expr.as<Let>();
        internal_error << "Let\n";
    }

    internal_error << "Can't infer bounds, Expr type not handled\n";
    return std::pair<Expr, Expr>();
}

std::pair<Expr, Expr> merge_bounds(const std::pair<Expr, Expr> &bounds0,
                                   const std::pair<Expr, Expr> &bounds1) {
    return {simplify(min(bounds0.first, bounds1.first)),
            simplify(max(bounds0.second, bounds1.second))};
};

Expr add_let_expression(const Expr &expr,
                        const std::map<std::string, Expr> &let_var_mapping,
                        const std::vector<std::string> &let_variables) {
    // TODO: find a faster way to do this
    Expr ret = expr;
    ret = remove_let_definitions(ret);
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto &let_variable : let_variables) {
            if (has_variable(ret, let_variable) &&
                    !has_let_defined(ret, let_variable)) {
                auto value = let_var_mapping.find(let_variable)->second;
                ret = Let::make(let_variable, value, ret);
                changed = true;
            }
        }
    }
    return ret;
}

/** Gather the expression DAG and sort them in topological order
 */
class ExpressionSorter : public IRGraphVisitor {
public:
    using IRGraphVisitor::visit;
    using IRGraphVisitor::include;

    std::vector<Expr> sort(const Expr &expr);

    void visit(const Call *op);
protected:
    void include(const Expr &e);
private:
    std::vector<Expr> expr_list;
};

std::vector<Expr> ExpressionSorter::sort(const Expr &e) {
    visited.clear();
    expr_list.clear();
    e.accept(this);
    expr_list.push_back(e);
    return expr_list;
}

void ExpressionSorter::visit(const Call *op) {
    // No point visiting the arguments of a Halide func or an image
    if (op->call_type == Call::Halide || op->call_type == Call::Image) {
        return;
    }

    for (size_t i = 0; i < op->args.size(); i++) {
        include(op->args[i]);
    }
}


void ExpressionSorter::include(const Expr &e) {
    if (visited.count(e.get())) {
        return;
    } else {
        visited.insert(e.get());
        e.accept(this);
        expr_list.push_back(e);
        return;
    }
}

std::vector<Expr> sort_expressions(const Expr &expr) {
	ExpressionSorter sorter;
	return sorter.sort(expr);
}


/**
 *  Visit function calls and determine their bounds.
 *  So when we do f(x, y) = ... we know what the loop bounds are
 */
class BoundsInferencer : public IRVisitor {
public:
    using IRVisitor::visit;

    void inference(const Func &func);
    void internal_inference(const Func &func);

    void visit(const Call *op);
    void visit(const Let *op);

    std::map<FuncKey, RDom> get_func_bounds() const {
        std::map<FuncKey, RDom> ret;
        // Convert to an Rdom
        for(auto b: func_bounds) { 
            debug(0) << "Computed bounds for " << b.first.first << "[" << b.first.second << "]" << ":\n";
            FuncBounds min_extent_bounds;
            min_extent_bounds.reserve(b.second.size());
            for (int i = 0; i < (int)b.second.size(); ++i) {
                Expr lower_bound = simplify(b.second[i].first);
                Expr extent = simplify(b.second[i].second - lower_bound+1);
                min_extent_bounds.push_back(std::make_pair(lower_bound, extent));
                debug(0) << "  arg" << i << " ("  << lower_bound << ", " << extent << ")\n";
            }
            ret[b.first] = RDom(min_extent_bounds);
        }
        return ret;
    }

    void update_func_bound();

private:
    int recursion_depth;
    std::map<FuncKey, FuncBounds> func_bounds;
    FuncKey current_func_key;
    std::vector<Var> current_args;
    RDom current_bounds;
    Scope<Expr> scope;
};

void BoundsInferencer::inference(const Func &func) {
    // initialization
    func_bounds.clear();
    recursion_depth = 0;
    current_func_key = FuncKey{"", -1};
    current_args.clear();
    current_bounds = RDom();

    FuncBounds bounds;
    for (int i = 0; i < (int)func.args().size(); i++) {
        bounds.push_back({1, 1});
    }
    // assume the output has size 1
    func_bounds[FuncKey{func.name(), func.num_update_definitions() - 1}] =
        bounds;

    internal_inference(func);
}

void BoundsInferencer::internal_inference(const Func &func) {
    FuncKey previous_func_key = current_func_key;
    RDom previous_bounds = current_bounds;
    std::vector<Var> previous_args = current_args;

    // Traverse from the last update to first
    for (int update_id = func.num_update_definitions() - 1; update_id >= -1; update_id--) {
        current_func_key = FuncKey{func.name(), update_id};
        current_bounds = RDom(func_bounds[current_func_key]);
        current_args = func.args();
        if (update_id >= 0) {
            func.update_value(update_id).accept(this);
        } else {
            func.value().accept(this);
        }

        if (update_id >= 0) {
            // Update function bounds for next update
            FuncBounds bounds = func_bounds[current_func_key];
            FuncKey next_key{func.name(), current_func_key.second - 1};
            if (func_bounds.find(next_key) != func_bounds.end()) {
                FuncBounds next_bounds = func_bounds[next_key];
                assert(bounds.size() == next_bounds.size());
                for (int i = 0; i < (int)bounds.size(); i++) {
                    bounds[i] = merge_bounds(bounds[i], next_bounds[i]);
                }
            }
            func_bounds[next_key] = bounds;
        }
    }

    current_func_key = previous_func_key;
    current_args = previous_args;
    current_bounds = previous_bounds;
}

void BoundsInferencer::visit(const Call *op) {
    if (op->call_type == Call::Halide) {
        Func func(Function(op->func));
        // Avoid implicit functions
        // TODO: FIX THIS
        if (func.args().size() > 0 && func.args()[0].is_implicit()) {
            return;
        }

        FuncBounds arg_bounds;
        arg_bounds.reserve(op->args.size());
        for (int i = 0; i < (int)op->args.size(); i++) {
            std::pair<Expr, Expr> min_max_bounds =
                get_min_max_bounds(op->args[i], current_args, current_bounds, i, scope);
            arg_bounds.push_back(min_max_bounds);
        }

        // Update function bounds
        FuncKey key = current_func_key.first == func.name() ?
            FuncKey{func.name(), current_func_key.second - 1} :
            FuncKey{func.name(), func.num_update_definitions() - 1};

        if (func_bounds.find(key) != func_bounds.end()) {
            FuncBounds prev_bounds = func_bounds[key];
            assert(arg_bounds.size() == prev_bounds.size());
            for (int i = 0; i < (int)arg_bounds.size(); i++) {
                arg_bounds[i] = merge_bounds(prev_bounds[i], arg_bounds[i]);
            }
        }

        func_bounds[key] = arg_bounds;

        // Don't recurse if the target is the same function
        if (current_func_key.first != func.name()) {
            recursion_depth += 1;
            internal_inference(func);
            recursion_depth -= 1;
        }
    }

    for (size_t i = 0; i < op->args.size(); i++) {
        op->args[i].accept(this);
    }
}

void BoundsInferencer::visit(const Let *op) {
    op->value.accept(this);
    scope.push(op->name, op->value);
    op->body.accept(this);
    scope.pop(op->name);
}

std::map<FuncKey, RDom> inference_bounds(const Func &func) {
	BoundsInferencer inferencer;
	inferencer.inference(func);
	return inferencer.get_func_bounds();
}

std::vector<std::pair<Expr, Expr>> rdom_to_vector(const RDom &bounds) {
    std::vector<std::pair<Expr, Expr>> ret;
    ret.reserve(bounds.domain().domain().size());
    for (const auto &rvar : bounds.domain().domain()) {
        ret.push_back({rvar.min, rvar.extent});
    }
    return ret;
}

Func set_boundary_zero(Func func, const RDom &bounds) {
    // Set up boundary condition
    Expr out_of_bounds = cast<bool>(false);
    for (size_t i = 0; i < bounds.domain().domain().size(); i++) {
        Var arg_var = func.args()[i];
        Expr min = bounds[i].min();
        Expr extent = bounds[i].extent();

        internal_assert(min.defined() && extent.defined());

        out_of_bounds = (out_of_bounds ||
                         arg_var < min ||
                         arg_var >= min + extent);
    }

    func(func.args()) = select(out_of_bounds, 0.f, func(func.args()));
    return func;
}

bool equal(const RDom &bounds0, const RDom &bounds1) {
    if (bounds0.domain().domain().size() != bounds1.domain().domain().size()) {
        return false;
    }
    for (int bid = 0; bid < (int)bounds0.domain().domain().size(); bid++) {
        if (!equal(bounds0[bid].min(), bounds1[bid].min()) ||
                !equal(bounds0[bid].extent(), bounds1[bid].extent())) {
            return false;
        }
    }
    return true;
}

std::vector<std::string> vars_to_strings(const std::vector<Var> &vars) {
    std::vector<std::string> ret;
    ret.reserve(vars.size());
    for (const auto &var : vars) {
        ret.push_back(var.name());
    }
    return ret;
}

} // namespace Internal
} // namespace Halide