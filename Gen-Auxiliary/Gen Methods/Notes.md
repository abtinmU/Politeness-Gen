# Basics

## The ways to represent a distribution:

1. x = {:x} ~ distribution(parameters)
Why use it? It gives a proper addres for each of our cases
Why not use it? It's a bit complex
2. x = {*} ~ distribution(parameters)
Why use it? It's a wildcard address. Easy to use. Random choices are imported directly into the caller’s trace
Why not use it? You can't use the same function twice
3. x = distribution(parameters)
Why use it? A syntactic sugar t make it look better. Desugars to `"x = {:x} ~ distribution(parameters)"`
Why not use it? The random choices made  won't be traced at all


## Functions to work on the trace

- trace = Gen.simulate(model, (xs,))  # run the generative function and obtain its trace

- xs = Gen.get_args(trace)

- Gen.get_choices(trace)  # print the choice map

- xs[:slope]  # pull out individual values from a map using Julia’s subscripting syntax [...]

- trace[:slope]  # read the value of a random choice directly from the trace

- y = Gen.get_retval(trace)  # take out the return values

- y - trace[]  # syntactic sugar to get the return values

- observations = Gen.choicemap()  # generate an empty choice map and tie your values to it using syntax similar to `observations[(:y, 5)] = y (corresponding to the 5th element of your return values)` or `observations[:slope] = slope`

Definition: A choice map maps random choice addresses from the model to values from our data set

- trace[Pair(:z, :y)]  # a hierarchical address. more complex example: `trace[Pair(:a, Pair(:z, :y))]`

- trace[:z => :y]  # method 2 for a hierarchical address. more complex example: `trace[:a => :z => :y]`

## Modeling special notes

- julia `struct` and recursion make a great combination sometimes! The following is a part of Gen tutorial on Bayesian nonparametrics & trees: 

~~~julia
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
@gen function generate_segments(l::Float64, u::Float64)
    interval = Interval(l, u)
    if ({:isleaf} ~ bernoulli(0.7))
        value = ({:value} ~ normal(0, 1))
        return LeafNode(value, interval)
    else
        frac = ({:frac} ~ beta(2, 2))
        mid  = l + (u - l) * frac
        # Call generate_segments recursively!
        # Because we will call it twice -- one for the left 
        # child and one for the right child -- we use
        # addresses to distinguish the calls.
        left = ({:left} ~ generate_segments(l, mid))
        right = ({:right} ~ generate_segments(mid, u))
        return InternalNode(left, right, interval)
    end
end;
~~~

- easy way to get a uniform initial xs: `xs = collect(range(-5, stop=5, length=n))`

- push!(values, value)  # add a taken sample, `value`, to the `values` vector. Example use cases:

`push!(ys, {:data => i => :y} ~ normal(mu, std))`

`push!(ys, y)`


# Gen's Programmable Inference

## Some points:

`observations = Gen.choicemap()`

- Gen's generate() function takes 3 inputs: a model, a tuple of input arguments to the model, and a `ChoiceMap` representing observations (or constraints to satisfy). It returns a complete trace consistent with the observations, and an importance weight.

`(tr, _) = generate(model, (xs,), observations, 1000)`


## Basic Inference

### importance resampling: 

`(traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction, model_args::Tuple, observations::ChoiceMap, num_samples::Int; verbose=false)`

`(traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction, model_args::Tuple, observations::ChoiceMap, proposal::GenerativeFunction, proposal_args::Tuple, num_samples::Int; verbose=false)`

Importance sampling can be difficult to scale to more complex problems, because it is essentially “guessing and checking.” For example, if we run importance sampling with 1000 particles, the method would fail unless those 1000 proposed solutions (blind guesses, essentially) contained something close to the true answer. In complex problems, it is difficult to “guess” (or “propose”) an entire solution all at once.

### metropolis hastings:

- `metropolis_hastings` operator or `mh` function automatically adds the “accept/reject” check. So that inference programmers need only think about what sorts of updates might be useful to propose.

`(tr, did_accept) = mh(tr, select(:address1, :address2, ...))`

`(tr, did_accept) = mh(tr, custom_proposal, custom_proposal_args)`

- we will now see some `mh` usage and some sample design patterns for MCMC updates from the Gen tutorial:

#### Block Resimulation

Long story short, all we do in a single Block Resimulation step is to apply metropolis hastings on our trace and update each prameters of our model.


~~~julia
# This example, assumes we have a pre-defined bayesian linear model that in addition to a `slope` and `intercept`, also decides on whether or not a point is a outlier (along with some probability).

# Perform a single block resimulation update of a trace.
function block_resimulation_update(tr)
    # Block 1: Update the line's parameters
    line_params = select(:noise, :slope, :intercept)
    (tr, _) = mh(tr, line_params)
    
    # Blocks 2-N+1: Update the outlier classifications
    (xs,) = get_args(tr)
    n = length(xs)
    for i=1:n
        (tr, _) = mh(tr, select(:data => i => :is_outlier))
    end
    
    # Block N+2: Update the prob_outlier parameter
    (tr, _) = mh(tr, select(:prob_outlier))
    
    # Return the updated trace
    tr
end;


# All that’s left is to obtain an initial trace, and run our update in a loop for as long as we’d like:

function block_resimulation_inference(xs, ys, observations)
    observations = make_constraints(ys)
    (tr, _) = generate(regression_with_outliers, (xs,), observations)
    for iter=1:500
        tr = block_resimulation_update(tr)
    end
    tr
end;
~~~


#### Gaussian Drift

- In this update method, our updates are done by randomly perturbing the existing value

~~~julia
@gen function line_proposal(current_trace)
    slope ~ normal(current_trace[:slope], 0.5)
    intercept ~ normal(current_trace[:intercept], 0.5)
end;

function gaussian_drift_update(tr)
    # Gaussian drift on line params
    (tr, _) = mh(tr, line_proposal, ())
    
    # Block resimulation: Update the outlier classifications
    (xs,) = get_args(tr)
    n = length(xs)
    for i=1:n
        (tr, _) = mh(tr, select(:data => i => :is_outlier))
    end
    
    # Block resimulation: Update the prob_outlier parameter
    (tr, w) = mh(tr, select(:prob_outlier))
    (tr, w) = mh(tr, select(:noise))
    tr
end;
~~~

#### Improving the inference: Heuristics

- This technique reminds me of some feature extraction/fine-tuning methods in machine learnnig and bootstrap/jackknife methods in statistics. To summarize, here we take small random subset of our points. Then, we do least-squares linear regression & find the fitted line for those points. And then, we check the quakity (distance) of this fitted line for the rest of our points. This gives us a 'score' for that line.

After doing this process multiple times we get a list of possible regression lines. We select the line with the best score.

~~~julia
struct RANSACParams
    """the number of random subsets to try"""
    iters::Int

    """the number of points to use to construct a hypothesis"""
    subset_size::Int

    """the error threshold below which a datum is considered an inlier"""
    eps::Float64
    
    function RANSACParams(iters, subset_size, eps)
        if iters < 1
            error("iters < 1")
        end
        new(iters, subset_size, eps)
    end
end


function ransac(xs::Vector{Float64}, ys::Vector{Float64}, params::RANSACParams)
    best_num_inliers::Int = -1
    best_slope::Float64 = NaN
    best_intercept::Float64 = NaN
    for i=1:params.iters
        # select a random subset of points
        rand_ind = StatsBase.sample(1:length(xs), params.subset_size, replace=false)
        subset_xs = xs[rand_ind]
        subset_ys = ys[rand_ind]
        
        # estimate slope and intercept using least squares
        A = hcat(subset_xs, ones(length(subset_xs)))
        slope, intercept = A \ subset_ys # use backslash operator for least sq soln
        
        ypred = intercept .+ slope * xs

        # count the number of inliers for this (slope, intercept) hypothesis
        inliers = abs.(ys - ypred) .< params.eps
        num_inliers = sum(inliers)

        if num_inliers > best_num_inliers
            best_slope, best_intercept = slope, intercept
            best_num_inliers = num_inliers
        end
    end

    # return the hypothesis that resulted in the most inliers
    (best_slope, best_intercept)
end;

# We can now wrap it in a Gen proposal that calls out to RANSAC, then samples a slope and intercept near the one it proposed.

@gen function ransac_proposal(prev_trace, xs, ys)
    (slope_guess, intercept_guess) = ransac(xs, ys, RANSACParams(10, 3, 1.))
    slope ~ normal(slope_guess, 0.1)
    intercept ~ normal(intercept_guess, 1.0)
end;
~~~

# Introducing a **trainable** and **positive**,  parameter to an @Gen model  for training with *unconstrained parameters*: 

We use `@param log_...` macro: @param add the trainable pareter and the log version makes sure to keep it as a positive value after taking its exponent.

In the following, we can observe a sample such case that has the `score_high` parameter as a non-trainable constant:

~~~julia
@gen function custom_dest_proposal(measurements::Vector{Point}, scene::Scene)

    score_high = 5.
    
    x_first = measurements[1].x
    x_last = measurements[end].x
    y_first = measurements[1].y
    y_last = measurements[end].y
    
    # sample dest_x
    x_probs = compute_bin_probs(num_x_bins, scene.xmin, scene.xmax, x_first, x_last, score_high)
    x_bounds = collect(range(scene.xmin, stop=scene.xmax, length=num_x_bins+1))
    dest_x ~ piecewise_uniform(x_bounds, x_probs)
    
    # sample dest_y
    y_probs = compute_bin_probs(num_y_bins, scene.ymin, scene.ymax, y_first, y_last, score_high)
    y_bounds = collect(range(scene.ymin, stop=scene.ymax, length=num_y_bins+1))
    dest_y ~ piecewise_uniform(y_bounds, y_probs)
    
    return nothing
end;
~~~

Now, we turn `score_high` to a trainable version. Take note that we need `score_high` to stay a positive value during the training process.

~~~julia
@gen function custom_dest_proposal_trainable(measurements::Vector{Point}, scene::Scene)

    # making score_high to be a trainable parameter from
    @param log_score_high::Float64
    
    x_first = measurements[1].x
    x_last = measurements[end].x
    y_first = measurements[1].y
    y_last = measurements[end].y
    
    # sample dest_x
    x_probs = compute_bin_probs(num_x_bins, scene.xmin, scene.xmax, x_first, x_last, exp(log_score_high))
    x_bounds = collect(range(scene.xmin, stop=scene.xmax, length=num_x_bins+1))
    dest_x ~ piecewise_uniform(x_bounds, x_probs)
    
    # sample dest_y
    y_probs = compute_bin_probs(num_y_bins, scene.ymin, scene.ymax, y_first, y_last, exp(log_score_high))
    y_bounds = collect(range(scene.ymin, stop=scene.ymax, length=num_y_bins+1))
    dest_y ~ piecewise_uniform(y_bounds, y_probs)
    
    return nothing
end;
~~~

















