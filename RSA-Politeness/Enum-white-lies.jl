# -*- coding: utf-8 -*-
"""
The RSA Model of Politeness Using Enumeration
The case study of white lies
"""

import Pkg
Pkg.add("Gen")
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("StatsPlots")
Pkg.add("Distributions")

#import Random, Logging
using Gen, Plots, Distributions, StatsPlots
using StatsBase: mean, countmap
using LinearAlgebra

########################
### Literal Listener ###
########################

# Define state and utterance space
states = [1, 2, 3, 4, 5]
utterances = ["terrible", "bad", "okay", "good", "amazing"]
key_mapping = Dict("terrible" => 1, "bad" => 2, "okay" => 3, "good" => 4, "amazing" => 5)

# Literal semantics represented as probabilities
literalSemantics = Dict(
    "terrible" => [0.95, 0.85, 0.02, 0.02, 0.02],
    "bad" => [0.85, 0.95, 0.02, 0.02, 0.02],
    "okay" => [0.02, 0.25, 0.95, 0.65, 0.35],
    "good" => [0.02, 0.05, 0.55, 0.95, 0.93],
    "amazing" => [0.02, 0.02, 0.02, 0.65, 0.95]
)

prob(utterance::String, state::Int64) = literalSemantics[utterance][state]
@dist meaning(utterance::String, state::Int64) = bernoulli(prob(utterance, state))

@dist uniformDraw(vector::Vector, vectorProbs::Vector{Float64}) = vector[categorical(vectorProbs)]
uniformProbs(vector::Vector) = fill(1 / length(vector), length(vector))
stateProbs = uniformProbs(states)
utterancesProbs = uniformProbs(utterances)

@gen function literalListener(stateProbs::Vector{Float64}, utterance::String)

    state = @trace(uniformDraw(states, stateProbs), :state)
    m = @trace(meaning(utterance, state), :m)
    return state

end

"""
    enum_inference(model, model_args, observations, latent_addrs, latent_values)

Runs enumerative Bayesian inference for a `model` parameterized by `model_args`,
conditioned on the `observations`. Given a list of `latent_addrs` and the
a list of corresponding `latent_values` that each latent variable can take on,
we enumerate over all possible settings of the latent variables.

Returns a named tuple with the following fields:
- `traces`: An array of execution traces for each combination of latent values.
- `logprobs`: An array of log probabilities for each trace.
- `latent_logprobs`: A dictionary of log posterior probabilities per latent.
- `latent_probs`: A dictionary of posterior probabilities per latent.
- `lml`: The log marginal likelihood of the observations.

Source: The implementation of Cooperative Language-Guided Inverse Plan Search (CLIPS)
by MIT's Probabilistic Computing Lab: https://github.com/probcomp/CLIPS.jl
"""
function enum_inference(
    model::GenerativeFunction, model_args::Tuple,
    observations::ChoiceMap, latent_addrs, latent_values
)
    @assert length(latent_addrs) == length(latent_values)
    # Construct iterator over combinations of latent values
    latents_iter = Iterators.product(latent_values...)
    # Generate a trace for each possible combination of latent values
    traces = map(latents_iter) do latents
        constraints = choicemap()
        for (addr, val) in zip(latent_addrs, latents)
            constraints[addr] = val
        end
        constraints = merge(constraints, observations)
        tr, _ = generate(model, model_args, constraints)
        return tr
    end
    # Compute the log probability of each trace
    logprobs = map(Gen.get_score, traces)
    # Compute the log marginal likelihood of the observations
    lml = logsumexp(logprobs)
    # Compute the (marginal) posterior probabilities for each latent variable
    latent_logprobs = Dict(
        addr => ([logsumexp(lps) for lps in eachslice(logprobs, dims=i)] .- lml)
        for (i, addr) in enumerate(latent_addrs)
    )
    latent_probs = Dict(addr => exp.(lp) for (addr, lp) in latent_logprobs)
    return (
        traces = traces,
        logprobs = logprobs,
        latent_logprobs = latent_logprobs,
        latent_probs = latent_probs,
        latent_addrs = latent_addrs,
        lml = lml
    )
end

# Define observed actions
utterance = "good"  # The observed utterance
observations = choicemap((:m, 1))  # Observing that m is true

# Run inference by enumerating over all possible states
# results = enum_inference(
#     literalListener,
#     (utterancesProbs, utterance),
#     observations,
#     (:state, ),
#     (states, )
# )

# Print the inferred state probabilities
# println("Inferred state probabilities given that the utterance was `good` and m=true:")
# for (state, prob) in zip([1, 2, 3, 4, 5], results.latent_probs[:state])
#     println("State $state: Probability $prob")
# end

# bar(states, results.latent_probs[:state], xlabel="States", ylabel="Proportions", legend=false)

##########################
### Pragmatic Speaker ###
##########################

lambda_ = 1.25
social(proportions::Dict, valueFunctionLambda) = sum(key * value for (key, value) in proportions) * valueFunctionLambda
state_logProb(L0_post::Dict, state::Int64) = log(L0_post[state])

# The (deterministic) generative function for the pragmatic speaker
## This version could be improved later ##
@gen function Speaker1(state::Int, utterance::String, phi::Float64)
    alpha_ = 10
    L0_post = enum_inference(
                  literalListener,
                  (utterancesProbs, utterance),
                  observations,
                  (:state, ),
                  (states, )
              )
    L0 = Dict(key => value for (key, value) in zip(states, L0_post.latent_probs[:state]))
    utility_epistemic = state_logProb(L0, state)
    utility_social = social(L0, lambda_)
    speakerUtility = phi * utility_epistemic + (1 - phi) * utility_social
    return alpha_ * speakerUtility
end

function enumeration_S1(state::Int, phi::Float64, utterances::Vector{String})
    lambda_ = 1.25
    alpha_ = 10

    # Calculate utilities for each utterance
    utilities = Dict(utterance => Speaker1(state, utterance, phi) for utterance in utterances)

    # Convert utilities to probabilities using softmax
    max_utility = maximum(values(utilities))  # For numerical stability in softmax
    exp_utilities = [exp(utilities[u] - max_utility) for u in utterances]
    total_exp = sum(exp_utilities)
    probabilities = [eu / total_exp for eu in exp_utilities]

    return Dict(zip(utterances, probabilities))
end

state = 1
phi = 0.99
utterances = ["terrible", "bad", "okay", "good", "amazing"]

# Calling the enumeration algorithm
S1_post = enumeration_S1(state, phi, utterances)
# Extracting the probabilities of utterances in the original order
probabilities = [S1_post[utterance] for utterance in utterances]

bar(utterances, probabilities, xlabel="States", ylabel="Proportions", legend=false)

##########################
### Pragmatic Listener ###
##########################

# Save S1's probabilities for different states and phis for fast later access

phiVals = collect(0.05:0.05:0.95)
phiProbs = uniformProbs(phiVals)
utterances = ["terrible", "bad", "okay", "good", "amazing"]

S1_posterior_map = Dict((state, phi) => enumeration_S1(state, phi, utterances) for state in states for phi in phiVals)

@gen function pragmaticListener(utterance)

    state = @trace(uniformDraw(states, stateProbs), :state)
    phi = @trace(uniformDraw(phiVals, phiProbs), :phi)
    S1Dict = S1_posterior_map[(state, phi)]
    S1 = @trace(uniformDraw(collect(keys(S1Dict)), collect(values(S1Dict))), (:S1, utterance))

    return [state, phi]
end

## This needs further improvement. I leave it at this for now and write the new version soon using Gen ##
function enumerate_L1(utterance, states, stateProbs, phiVals, phiProbs, S1_posterior_map)
    # Dictionary to store the joint log probabilities for each (state, phi)
    log_probs = Dict()

    # Iterate over all states and phi values
    for state in states
        for phi in phiVals
            # Retrieve the probability of the utterance given state and phi from the precomputed S1 model
            prob_utterance_given_state_phi = get(S1_posterior_map[(state, phi)], utterance, 0)
            log_prob_utterance_given_state_phi = prob_utterance_given_state_phi > 0 ? log(prob_utterance_given_state_phi) : -Inf

            # Calculate the total log probability including the priors
            log_prob = log(stateProbs[findfirst(==(state), states)]) +
                       log(phiProbs[findfirst(==(phi), phiVals)]) +
                       log_prob_utterance_given_state_phi

            # Store the log probability
            log_probs[(state, phi)] = log_prob
        end
    end

    # Normalize the log probabilities to form a probability distribution
    max_log_prob = maximum(values(log_probs))
    probs = Dict(k => exp(v - max_log_prob) for (k, v) in log_probs)
    total_prob = sum(values(probs))
    normalized_probs = Dict(k => v / total_prob for (k, v) in probs)

    return normalized_probs
end

utterance = "good"  # The observed utterance
L1_post = enumerate_L1(utterance, states, stateProbs, phiVals, phiProbs, S1_posterior_map)

# Calculate the marginal probabilities
marginal_states = Dict(state => 0.0 for (state, _) in keys(L1_post))
for ((state, phi), prob) in L1_post
    marginal_states[state] += prob
end

marginal_phis = Dict(phi => 0.0 for (_, phi) in keys(L1_post))
for ((state, phi), prob) in L1_post
    marginal_phis[phi] += prob
end

state_probabilities = [marginal_states[state] for state in states]
println("Expected state: ", dot(state_probabilities, states))
bar(states, state_probabilities, xlabel="States", ylabel="Proportions", legend=false)

phi_probabilities = [marginal_phis[phi] for phi in phiVals]
println("Expected phi: ", dot(phi_probabilities, phiVals))
bar(phiVals, phi_probabilities, xlabel="Phis", ylabel="Proportions", legend=false)