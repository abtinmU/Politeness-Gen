module GenExerciseUtils

# These functions was taken directly from the Gen Tutorial. Some were modified by me to a better format.

function render_trace(trace; show_data=true)
    
    xs, = get_args(trace)
    
    xmin = minimum(xs)
    xmax = maximum(xs)

    y = get_retval(trace)
    
    test_xs = collect(range(-5, stop=5, length=1000))
    fig = plot(test_xs, map(y, test_xs), color="black", alpha=0.5, label=nothing,
                xlim=(xmin, xmax), ylim=(xmin, xmax))

    if show_data
        ys = [trace[(:y, i)] for i=1:length(xs)]
        
        scatter!(xs, ys, c="black", label=nothing)
    end
    
    return fig
end;

function do_inference(model, xs, ys, amount_of_computation)
    
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end
    
    (trace, _) = Gen.importance_resampling(model, (xs,), observations, amount_of_computation);
    return trace
end;

function overlay(renderer, traces; same_data=true, args...)
    fig = renderer(traces[1], show_data=true, args...)
    
    xs, = get_args(traces[1])
    xmin = minimum(xs)
    xmax = maximum(xs)

    for i=2:length(traces)
        y = get_retval(traces[i])
        test_xs = collect(range(-5, stop=5, length=1000))
        fig = plot!(test_xs, map(y, test_xs), color="black", alpha=0.5, label=nothing,
                    xlim=(xmin, xmax), ylim=(xmin, xmax))
    end
    return fig
end;

function predict_new_data(model, trace, new_xs::Vector{Float64}, param_addrs)
  
    constraints = Gen.choicemap()
    for addr in param_addrs
        constraints[addr] = trace[addr]
    end
    
    (new_trace, _) = Gen.generate(model, (new_xs,), constraints)
    
    ys = [new_trace[(:y, i)] for i=1:length(new_xs)]
    return ys
end;

function infer_and_predict(model, xs, ys, new_xs, param_addrs, num_traces, amount_of_computation)
    pred_ys = []
    for i=1:num_traces
        trace = do_inference(model, xs, ys, amount_of_computation)
        push!(pred_ys, predict_new_data(model, trace, new_xs, param_addrs))
    end
    pred_ys
end;

function plot_predictions(xs, ys, new_xs, pred_ys; title="predictions")
    fig = scatter(xs, ys, color="red", label="observed data", title=title)
    for (i, pred_ys_single) in enumerate(pred_ys)
        scatter!(new_xs, pred_ys_single, color="black", alpha=0.1, label=i == 1 ? "predictions" : nothing)
    end
    return fig
end;

struct Interval
    l::Float64
    u::Float64
end
abstract type Node end
    
struct InternalNode <: Node
    left::Node
    right::Node
    interval::Interval
end

struct LeafNode <: Node
    value::Float64
    interval::Interval
end

function render_node!(node::LeafNode)
    plot!([node.interval.l, node.interval.u], [node.value, node.value], label=nothing, linewidth=5)
end

function render_node!(node::InternalNode)
    render_node!(node.left)
    render_node!(node.right)
end;
function render_segments_trace(trace; xlim=(0,1))
    node = get_retval(trace)
    fig = plot(xlim=xlim, ylim=(-3, 3))
    render_node!(node)
    return fig
end;

@gen function regression_with_outliers(xs::Vector{<:Real})

    slope ~ normal(0, 2)
    intercept ~ normal(0, 2)
    noise ~ gamma(1, 1)
    prob_outlier ~ uniform(0, 1)

    n = length(xs)
    ys = Float64[]
    
    for i = 1:n

        if ({:data => i => :is_outlier} ~ bernoulli(prob_outlier))
            (mu, std) = (0., 10.)
        else
            (mu, std) = (xs[i] * slope + intercept, noise)
        end

        push!(ys, {:data => i => :y} ~ normal(mu, std))
    end
    ys
end;

function extract_data_from_trace(trace)
    xs, = get_args(trace)
    ys = get_retval(trace)
    
    is_outliers = Bool[]
    for i in 1:length(xs)
        push!(is_outliers, getindex(trace, :data => i => :is_outlier))
    end
    
    return xs, ys, is_outliers
end

function plot_trace_with_data(trace)
    plot(xlims=(-5, 5), ylims=(-20, 20))
    
    xs, ys, is_outliers = extract_data_from_trace(trace)
    
    scatter!(xs[.!is_outliers], ys[.!is_outliers], color=:blue, legend=false)
    scatter!(xs[is_outliers], ys[is_outliers], color=:red, legend=false)
    
    noise = trace[:noise]
    regression_line_y = trace[:intercept] .+ trace[:slope] .* xs
    upper_shade_y = regression_line_y .+ noise
    lower_shade_y = regression_line_y .- noise
    plot!(xs, upper_shade_y, fillrange=lower_shade_y, fillalpha=0.5, color=:darkgrey, legend=false)
    
    plot!([minimum(xs), maximum(xs)], [10, 10], fillrange=[-10, -10], fillalpha=0.2, color=:lightgrey, legend=false)
    
    plot!(xs, regression_line_y, color=:black, linestyle=:solid, legend=false)
end

end # module GenExerciseUtils
