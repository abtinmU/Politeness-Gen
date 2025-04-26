"""
The RSA Model of Politeness Using Enumeration
The case study of white lies

Version 2: Using Gen's new `enumerative_inference` method
"""

########################
### Prepare Packages ###
########################

import Pkg
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("StatsPlots")
Pkg.add("Distributions")
Pkg.add("StatsFuns")

#import Random, Logging
using Plots, Distributions, StatsPlots, LinearAlgebra, StatsFuns, Printf
using StatsBase: mean, countmap
using Distributions: Categorical

"""
Install Gen's Latest Repo Version
The current Gen.jl stable release (v0.4.7) does not ship with `enumerative_inference` in its API. 
We need to clone the latest Gen.jl repo and proceed with that version.
"""

; git clone https://github.com/probcomp/Gen.jl.git /content/Gen.jl

Pkg.develop(path="/content/Gen.jl")
@show hasproperty(Gen, :enumerative_inference)
]

########################
### Literal Listener ###
########################

using Gen

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

# state‐grid iterator
grid_iter = choice_vol_grid((:state, states))

# function to get posterior for one utterance u
function L0_posterior(u::String)
    (traces, log_weights, lml_est) = enumerative_inference(
        literalListener,
        (stateProbs, u),   # stateProbs + single utterance
        choicemap((:m, 1)),# observed m = true at address :m
        grid_iter          # positional iterator for :state
    )
    return Dict(zip(states, exp.(log_weights))), lml_est
end

for u in utterances
    posterior, lml = L0_posterior(u)
    println("Utterance = \"$u\" (log-marginal = $(round(lml, digits=3))):")
    for s in states
        @printf("  state=%d → %.3f\n", s, posterior[s])
    end
    println()
end

# for utterance = "good"
posterior_on_state, lml = L0_posterior("good")
posterior_vec = [posterior_on_state[s] for s in states]
bar(
  states, posterior_vec;
  xlabel="States",
  ylabel="P(state | \"good\")",
  legend=false,
  title="L0 posterior via Gen Enumeration"
)


##########################
### Pragmatic Speaker ###
##########################

# grid‐iterators for enumeration
state_grid = choice_vol_grid((:state, states))
utter_grid = choice_vol_grid((:utter, utterances))

# S1 utility function defined as a simple Julia function
const λ = 1.25
function S1_util(utt::String, st::Int, φ::Float64)
    α = 10.0
    L0, lml = L0_posterior(utt)
    u_epi = log(L0[st])
    u_soc = sum(k*v for (k,v) in L0) * λ
    return α * (φ * u_epi + (1 - φ) * u_soc)
end

# utilities → normalized choice‐probs
function speaker_log_probs(
    utts::Vector{String}, priors::Vector{Float64},
    util_fn::Function, util_args::Tuple
)
    adjusted = Float64[]
    for (u, p) in zip(utts, priors)
        util = util_fn(u, util_args...)
        push!(adjusted, exp(log(p) + util))
    end
    return adjusted ./ sum(adjusted)
end

# Pragmatic speaker model
@gen function speaker1(st::Int, φ::Float64)
    S1p = speaker_log_probs(
        utterances, utterancesProbs,
        S1_util, (st, φ)
    )
    utt = @trace(uniformDraw(utterances, S1p), :utter)
    return utt
end

# Enumerative inference on speaker1
function utterProbsDict(state::Int, φ::Float64)
    # enumerate over :utter to get exact S₁ posterior for this (state, φ)
    (tr_s1, logw_s1, lml_s1) = enumerative_inference(
        speaker1,
        (state, φ),   # arguments: state and phi
        choicemap(),  # no observations
        utter_grid    # enumerate over utterance choices
    )
    return tr_s1, Dict(zip(utterances, exp.(logw_s1))), lml_s1
end
_, logw_dict, lml_s1 = utterProbsDict(1, 0.99) # state=1, φ=0.99

posterior_s1 = [logw_dict[u] for u in utterances]

println("Log‐marginal likelihood (speaker1): ", round(lml_s1, digits=3))
println("Posterior P(utterance | state=1, φ=0.99):")
for (u, p) in logw_dict
    @printf("  %-8s → %.3f\n", u, p)
end

bar(
    utterances, posterior_s1;
    xlabel="Utterance",
    ylabel="P(utterance)",
    title="Pragmatic Speaker Posterior",
    legend=false
)


##########################
### Pragmatic Listener ###
##########################

phiVals = collect(0.05:0.05:0.95)
phiProbs = uniformProbs(phiVals)
utterances = ["terrible", "bad", "okay", "good", "amazing"]

# A Gen model whose only latent choices are :state, :phi, and :utter
@gen function pragmaticListener()
  # 1) draw state and phi from their priors
  s  = @trace(uniformDraw(states, stateProbs), :state)
  φ  = @trace(uniformDraw(phiVals,  phiProbs),   :phi)

  dict_u = utterProbsDict(s, φ)[2]    #  the S1’s distribution over utterances
  p_u = [ dict_u[u] for u in utterances ]

  utt = @trace(uniformDraw(utterances, p_u), :utter)
  return (state=s, phi=φ)
end

joint_grid = choice_vol_grid(
  (:state, states),
  (:phi,   phiVals)
)

# enumerative inference, conditioning on the observed utterance
observed = "good"
(tr_pl, logw_pl, lml_pl) = enumerative_inference(
  pragmaticListener,
  (),
  choicemap(:utter => observed),
  joint_grid
)

# normalized weights
weights = exp.(logw_pl)

state_post = Dict(
  s => sum(weights[i] for (i,(cm,_)) in enumerate(joint_grid) if cm[:state] == s)
  for s in states
)
phi_post = Dict(
  φ => sum(weights[i] for (i,(cm,_)) in enumerate(joint_grid) if cm[:phi]   == φ)
  for φ in phiVals
)

println("Log-marginal likelihood (pragmaticListener): ", round(lml_pl, digits=3))

println("\nPosterior P(state | utterance=\"$observed\"):")
for s in states
    @printf("  state=%d → %.3f\n", s, state_post[s])
end

println("\nPosterior P(phi   | utterance=\"$observed\"):")
for φ in phiVals
    @printf("  φ=%.2f → %.3f\n", φ, phi_post[φ])
end

bar(states, [state_post[s] for s in states];
    xlabel="State", ylabel="P(state)",
    title="P(state|\"$observed\")", legend=false)

bar(phiVals, [phi_post[φ] for φ in phiVals];
    xlabel="φ", ylabel="P(φ)",
    title="P(φ|\"$observed\")", legend=false)