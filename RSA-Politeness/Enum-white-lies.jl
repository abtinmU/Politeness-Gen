"""
The RSA Model of Politeness Using Enumeration
The case study of white lies
"""

########################
### Prepare Packages ###
########################

import Pkg
Pkg.add("Gen")
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("StatsPlots")
Pkg.add("Distributions")
Pkg.add("StatsFuns")
#Pkg.add("Luxor")

#import Random, Logging
using Gen, Plots, Distributions, StatsPlots, LinearAlgebra, StatsFuns
using StatsBase: mean, countmap


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

Source: This implementation was taken and modified from the implementation
of Cooperative Language-Guided Inverse Plan Search (CLIPS) from MIT's
Probabilistic Computing Lab: https://github.com/probcomp/CLIPS.jl
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
# utterance = "good"  # The observed utterance
# observations = choicemap((:m, 1))  # Observing that m is true

# Run inference by enumerating over all possible states
# results = enum_inference(
#     literalListener,
#     (utterancesProbs, utterance),
#     observations,
#     (:state, ),  # latent addresses
#     (states, )
# )

# Print inferred state probabilities
# println("Inferred state probabilities given that the utterance was `good` and m=true:")
# for (state, prob) in zip(states, results.latent_probs[:state])
#     println("State $state: Probability $prob")
# end

# bar(states, results.latent_probs[:state], xlabel="States", ylabel="Proportions", legend=false)


##########################
### Pragmatic Speaker ###
##########################

lambda_ = 1.25
social(proportions::Dict, valueFunctionLambda) = sum(key * value for (key, value) in proportions) * valueFunctionLambda
state_logProb(L0_post::Dict, state::Int64) = log(L0_post[state])

@gen function S1_utility(utterance::String, state::Int, phi::Float64)
    alpha_ = 10
    L0_post = enum_inference(
                  literalListener,
                  (utterancesProbs, utterance),
                  choicemap((:m, 1)),
                  (:state, ),
                  (states, )
              )
    L0 = Dict(key => value for (key, value) in zip(states, L0_post.latent_probs[:state]))
    utility_epistemic = state_logProb(L0, state)
    utility_social = social(L0, lambda_)
    speakerUtility = phi * utility_epistemic + (1 - phi) * utility_social
    return alpha_ * speakerUtility
end

"""
    speaker_log_probs(utterances::Array{String}, utterProbs::Array{Float64}, speakerUtil::Gen.DynamicDSLFunction, utilArgs::Tuple)

Calculate the adjusted probabilities of each utterance based on their initial probabilities and utility.
# Arguments
- utterances: Array{String} - A list of possible utterances.
- utterProbs: Array{Float64} - A list containing the probability of each utterance, corresponding to the `utterances` array.
- utilArgs: Tuple - A tuple of input parameters to calculate the utility value of each utterance.
- speakerUtil: DynamicDSLFunction - A function that takes an utterance and other necessary arguments as input and returns the utility value for that utterance.
# Output
- newProbs: Array{Float64}: An array containing the adjusted probabilities of each utterance. The probabilities are adjusted by adding
the utility of each utterance to the its original log probability.
"""

function speaker_log_probs(utterances::Array{String}, utterProbs::Array{Float64}, speakerUtil::Gen.DynamicDSLFunction, utilArgs::Tuple)

    # Calculate utility-adjusted probabilities
    adjusted_probs = Float64[]  # Initialize an empty array to store results

    for (utterance, prob) in zip(utterances, utterProbs)
        utility = speakerUtil(utterance, utilArgs...)
        log_prob = log(prob)  # Compute log of the probability
        adjusted_prob = exp(log_prob + utility)  # Combine and exponentiate
        push!(adjusted_probs, adjusted_prob)  # Append to results array
    end

    # Normalize the probabilities to sum to 1
    total_sum = sum(adjusted_probs)
    normalized_probs = adjusted_probs ./ total_sum

    return normalized_probs
end

# Calculate speaker utility and plot the results
resulting_probs = speaker_log_probs(utterances, [0.2,0.2,0.2,0.2,0.2], S1_utility, (1, 0.99, ))

# Print the adjusted probabilities
println(resulting_probs)

# Now we define the pragmtic speaker's model

@gen function speaker1(state::Int64, phi::Float64)

    S1_probs = speaker_log_probs(utterances, [0.2,0.2,0.2,0.2,0.2], S1_utility, (state, phi, ))
    utterance = @trace(uniformDraw(utterances, S1_probs), :utter)
    return utterance

end

# Define observed actions
# observations = choicemap()  # We assume no observation for now

# # Run inference by enumerating over all possible states
# results = enum_inference(
#     speaker1,
#     (1, 0.99),
#     observations,
#     (:utter, ),  # latent addresses
#     (utterances, )
# )

# Print inferred utterance probabilities
# println("Inferred utterance probabilities given our observationns:")
# for (utterance, prob) in zip(utterances, results.latent_probs[:utter])
#     println("utterance $utterance: Probability $prob")
# end

# bar(utterances, results.latent_probs[:utter], xlabel="States", ylabel="Proportions", legend=false)


##########################
### Pragmatic Listener ###
##########################

# Save S1's probabilities for different states and phis for fast later access

utterProbsDict(state, phi, utterances) = Dict(utterance => prob for (utterance, prob) in zip(utterances, enum_inference(speaker1, (state, phi), observations, (:utter, ), (utterances, )).latent_probs[:utter]))
phiVals = collect(0.05:0.05:0.95)
phiProbs = uniformProbs(phiVals)
utterances = ["terrible", "bad", "okay", "good", "amazing"]
S1_posterior_map = Dict((state, phi) => utterProbsDict(state, phi, utterances) for state in states for phi in phiVals)

"""
    enumerate_L1(utterance::String, states::Vector{String}, stateProbs::Vector{Float64},
                 phiVals::Vector{String}, phiProbs::Vector{Float64},
                 S1_posterior_map::Dict{Tuple{String, String}, Dict{String, Float64}})

Calculate the posterior probabilities of states and φ (phi) values given an utterance, using an input dict of speaker1 posterior for different inuput values.

# Arguments
- `utterance::String`: A string representing the utterance for which the probabilities are to be calculated.
- `states::Vector{String}`: An array of possible states.
- `stateProbs::Vector{Float64}`: An array of prior probabilities corresponding to each state in `states`.
- `phiVals::Vector{String}`: An array of possible φ (phi) values.
- `phiProbs::Vector{Float64}`: An array of prior probabilities corresponding to each φ (phi) value in `phiVals`.
- `S1_posterior_map::Dict{Tuple{String, String}, Dict{String, Float64}}`: A dictionary mapping tuples of (state, phi) to another dictionary that maps utterances to their probabilities. This map is essential for computing the likelihood of the utterance given each state and φ combination.

# Returns
- `state_probs::Vector{Float64}`: An array containing the normalized probabilities of each state given the utterance, ordered according to the input array `states`.
- `phi_probs::Vector{Float64}`: An array containing the normalized probabilities of each φ (phi) value given the utterance, ordered according to the input array `phiVals`.

"""

function enumerate_L1(utterance::String, states::Vector{String}, stateProbs::Vector{Float64},
                 phiVals::Vector{String}, phiProbs::Vector{Float64},
                 S1_posterior_map::Dict{Tuple{String, String}, Dict{String, Float64}})
    # Initialize arrays to accumulate log probabilities, setting initial values to -Inf for log space addition
    state_log_probs = fill(-Inf, length(states))
    phi_log_probs = fill(-Inf, length(phiVals))

    # Iterate over all states and phi values
    for (i, state) in enumerate(states)
        for (j, phi) in enumerate(phiVals)
            # Retrieve the probability of the utterance given state and phi from the precomputed S1 model
            prob_utterance_given_state_phi = get(S1_posterior_map[(state, phi)], utterance, 0)
            log_prob_utterance_given_state_phi = prob_utterance_given_state_phi > 0 ? log(prob_utterance_given_state_phi) : -Inf

            # Calculate the total log probability including the priors
            log_prob = log(stateProbs[i]) + log(phiProbs[j]) + log_prob_utterance_given_state_phi

            # Accumulate log probabilities
            state_log_probs[i] = logsumexp([state_log_probs[i], log_prob])
            phi_log_probs[j] = logsumexp([phi_log_probs[j], log_prob])
        end
    end

    # Normalize the probabilities for states and phi values
    state_probs = normalize_log_probs(state_log_probs)
    phi_probs = normalize_log_probs(phi_log_probs)

    return state_probs, phi_probs
end

# Helper function to normalize log probabilities
function normalize_log_probs(log_probs)
    max_log_prob = logsumexp(log_probs)
    normalized_probs = exp.(log_probs .- max_log_prob)
    return normalized_probs ./ sum(normalized_probs)
end

# Now we define the pragmtic listener's model

@gen function pragmaticListener(utterance)

    updates_stateProbs, updates_phiProbs = enumerate_L1(utterance, states, stateProbs, phiVals, phiProbs, S1_posterior_map)
    state = @trace(uniformDraw(states, updates_stateProbs), :state)
    phi = @trace(uniformDraw(phiVals, updates_phiProbs), :phi)

    return [state, phi]
end

trace = Gen.simulate(pragmaticListener, ("good", ))
Gen.get_choices(trace)

# Define observed actions
observations = choicemap()  # We assume no observation here

# Run inference by enumerating over all possible states
results = enum_inference(
    pragmaticListener,
    ("good", ),
    observations,
    (:phi, :state, ),  # latent addresses
    (phiVals, states, )
)

# Print inferred utterance probabilities
println("Inferred state probabilities given our observationns:")
for (utterance, prob) in zip(states, results.latent_probs[:state])
    println("State $utterance: Probability $prob")
end

println("Inferred phi probabilities given our observationns:")
for (utterance, prob) in zip(phiVals, results.latent_probs[:phi])
    println("phi $utterance: Probability $prob")
end

bar(states, results.latent_probs[:state], xlabel="States", ylabel="Proportions", legend=false)

bar(phiVals, L1_post, xlabel="Phis", ylabel="Proportions", legend=false)