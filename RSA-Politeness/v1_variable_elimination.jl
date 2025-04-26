"""
The RSA Model of Politeness Using Enumeration
The case study of white lies

Version 1: Using VE‐based inference by `GenVariableElimination.jl`

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
Pkg.add("PyCall")
Pkg.add(url="https://github.com/probcomp/GenVariableElimination.jl")

using Gen, Plots, Distributions, StatsPlots, LinearAlgebra, StatsFuns, Printf
using StatsBase: mean, countmap
using Distributions: Categorical
using GenVariableElimination: compile_trace_to_factor_graph, variable_elimination,
                             Latent, Observation, conditional_dist, factor_graph_analysis,
                             compile_trace_to_factor_graph, draw_factor_graph
using StatsFuns: logsumexp


########################
### Literal Listener ###
########################

states     = [1,2,3,4,5]
utterances = ["terrible","bad","okay","good","amazing"]
literalSemantics = Dict(
  "terrible" => [0.95,0.85,0.02,0.02,0.02],
  "bad"      => [0.85,0.95,0.02,0.02,0.02],
  "okay"     => [0.02,0.25,0.95,0.65,0.35],
  "good"     => [0.02,0.05,0.55,0.95,0.93],
  "amazing"  => [0.02,0.02,0.02,0.65,0.95]
)
stateProbs = fill(1/length(states), length(states))

@gen (static) function literalListener(stateProbs, utt::String)
  state ~ categorical(stateProbs)
  m ~ bernoulli(literalSemantics[utt][state])
  return state
end

"""
    ve_enum_inference(
      model, model_args,
      observations::ChoiceMap,
      latent_addrs::NTuple{N,Any},
      latent_values::NTuple{N,AbstractVector}
    ) where {N}

Exact posterior over discrete latents via variable elimination.
Returns `Dict(addr ⇒ posterior_vector)`.
"""
function ve_enum_inference(
    model::GenerativeFunction, model_args::Tuple,
    observations::ChoiceMap,
    latent_addrs::NTuple{N,Any},
    latent_values::NTuple{N,AbstractVector}
  ) where {N}

  # one constrained generate() to get a template trace
  tr, _ = generate(model, model_args, observations)

  # automatically extract domains & dependencies for SML
  _, latents, observations_meta = factor_graph_analysis(tr, latent_addrs)

  # compile and eliminate
  fg      = compile_trace_to_factor_graph(tr, latents, observations_meta)
  elimord = collect(latent_addrs)                   # heuristic_order(fg,:min_fill) might also work?
  ve_res  = variable_elimination(fg, elimord)

  # posteriors
  post = Dict{Any,Vector{Float64}}()
  dummy = Vector{Any}(undef, N)
  for (i, addr) in enumerate(latent_addrs)
    idx             = fg.addr_to_idx[addr]
    intermediate_fg = ve_res.intermediate_fgs[idx]
    post[addr]      = conditional_dist(intermediate_fg, dummy, addr)
  end

  return post
end

utterance   = "good"
observations = choicemap(:m => true)

results = ve_enum_inference(
  literalListener,
  (stateProbs, utterance),
  observations,
  (:state,),    # the one latent we need
  (states,)     # its domain
)

for (s,p) in zip(states, results[:state])
  @printf("  state %d → %.4f\n", s, p)
end

bar(
  states, results[:state],
  xlabel="States", ylabel="P(state | data)",
  legend=false, title="Exact posterior via VE"
)


##########################
### Pragmatic Speaker ###
##########################

function L0_posterior(utt::String)
  post = ve_enum_inference(
    literalListener,
    (stateProbs, utt),
    choicemap(:m => true),
    (:state,),
    (states,)
  )
  return post[:state]
end

L0_map = Dict(u => L0_posterior(u) for u in utterances)

# VE-based speaker model that conditions on state
lambda_ = 1.25
social(proportions, λ) = sum(k*v for (k,v) in zip(states, proportions)) * λ
@dist uniformDraw(vector::Vector, vectorProbs::Vector{Float64}) = vector[categorical(vectorProbs)]
uniformProbs(vector::Vector) = fill(1 / length(vector), length(vector))

@gen function speaker1_model(s::Int, φ::Float64)
  probs = Float64[]
  for u in utterances
    post = L0_map[u]
    ue   = log(post[s])
    us   = social(post, lambda_)
    push!(probs, exp(φ*ue + (1-φ)*us))
  end
  probs ./= sum(probs)
  utter ~ uniformDraw(utterances, probs)
  return utter
end

tr_sp, _   = generate(speaker1_model, (1, 0.99), choicemap())
latents_sp = Dict{Any,Latent}(:utter => Latent(utterances, []))
obs_meta_sp= Dict{Any,Observation}()

fg_sp      = compile_trace_to_factor_graph(tr_sp, latents_sp, obs_meta_sp)
ve_sp      = variable_elimination(fg_sp, [:utter])
dummy_sp   = Vector{Any}(undef,1)
post_utter = conditional_dist(ve_sp.intermediate_fgs[1], dummy_sp, :utter)

println("Exact P(utter | state=1, φ=0.99):")
for (u,p) in zip(utterances, post_utter)
  @printf("  %-8s → %.4f\n", u, p)
end

bar(utterances, post_utter, xlabel="utterance", ylabel="P", legend=false)


##########################
### Pragmatic Listener ###
##########################

phiVals = collect(0.05:0.05:0.95)
phiProbs = uniformProbs(phiVals)
utterances = ["terrible", "bad", "okay", "good", "amazing"]

prob(u,s) = literal[u][s]
@dist meaning(u::String,s::Int) = bernoulli(prob(u,s))
@dist catIdx(p)      = categorical(p)
@dist labCat(labels,p)= labels[catIdx(p)]

## L0  P(state | u , m=true)
@gen (static) function L0_listener(u::String)
    s ~ labCat(states, stateProbs)
    m ~ meaning(u, s)
    return s
end

alpha_ = 10
U = zeros(Float64, length(states), length(phiVals), length(utterances))
for (ui,u) in enumerate(utterances), (si,s) in enumerate(states), (pi,φ) in enumerate(phiVals)
    val = alpha_ * ( φ*log(L0_map[u][si])
                    + (1-φ)*(sum(k*v for (k,v) in zip(states,L0_map[u])) * lambda_) )
    U[si,pi,ui] = val
end

# softmax over utterances
for si in 1:length(states), pi in 1:length(phiVals)
    row = @view U[si,pi,:]
    row .-= maximum(row)
    row .= exp.(row)
    row ./= sum(row)
end

# Pragmatic listener
@gen (static) function L1_model(obsU::Int)
    sIdx ~ catIdx(stateProbs)
    pIdx ~ catIdx(phiProbs)
    uIdx ~ catIdx(@view U[sIdx,pIdx,:])
    return (sIdx, pIdx)
end

# utterance = "good"
obsU = findfirst(==("good"), utterances)
tr,_ = generate(L1_model, (obsU,), choicemap(:uIdx=>obsU))

lat = Dict{Any,Latent}(
    :sIdx => Latent(collect(1:length(states)),  []),
    :pIdx => Latent(collect(1:length(phiVals)), [])
)
obs = Dict{Any,Observation}(
    :uIdx => Observation([:sIdx, :pIdx])
)

fg = compile_trace_to_factor_graph(tr, lat, obs)

# marginal P(state | u)
ve_s  = variable_elimination(fg, [:pIdx, :sIdx])
i_s   = fg.addr_to_idx[:sIdx]
post_s = conditional_dist(ve_s.intermediate_fgs[i_s],
                         Vector{Any}(undef,2), :sIdx)

# marginal P(phi | u)
ve_p  = variable_elimination(fg, [:sIdx, :pIdx])
i_p   = fg.addr_to_idx[:pIdx]
post_p = conditional_dist(ve_p.intermediate_fgs[i_p],
                         Vector{Any}(undef,2), :pIdx)

println("Exact P(state | \"good\"):")
for (s,p) in zip(states, post_s)
    @printf("  %d → %.4f\n", s, p)
end

println("\nExact P(φ | \"good\"):")
for (i,φ) in enumerate(phiVals)
    @printf("  %.2f → %.4f\n", φ, post_p[i])
end

p_state = bar(
  states,
  post_s,
  xlabel="State",
  ylabel="P(state | \"good\")",
  legend=false,
  title="Posterior over states"
)

p_phi = bar(
  phiVals,
  post_p,
  xlabel="φ",
  ylabel="P(φ | \"good\")",
  legend=false,
  title="Posterior over φ"
)


############################################################
### generating GenVariableElimination.jl's factor graphs ###
############################################################

using PyCall
graphviz = pyimport("graphviz")
addr_to_name(addr) = replace(string(addr), ":" => "")

tr_L0, _ = generate(
  literalListener,
  (stateProbs, utterance),
  choicemap(:m => true),
)

_, latents_L0, obs_meta_L0 = factor_graph_analysis(tr_L0, (:state,))
fg_L0 = compile_trace_to_factor_graph(tr_L0, latents_L0, obs_meta_L0)
draw_factor_graph(fg_L0, graphviz, "L0", addr_to_name)

tr_sp, _ = generate(
  speaker1_model,
  (1, 0.99),
  choicemap()
)

latents_sp = Dict{Any,Latent}(:utter => Latent(utterances, []))
obs_meta_sp = Dict{Any,Observation}()
fg_sp = compile_trace_to_factor_graph(tr_sp, latents_sp, obs_meta_sp)
draw_factor_graph(fg_sp, graphviz, "S1", addr_to_name)

obsU = findfirst(==("good"), utterances)
tr_l1, _ = generate(
  L1_model,
  (obsU,),
  choicemap(:uIdx=>obsU)
)

lat_L1 = Dict{Any,Latent}(
  :sIdx => Latent(collect(1:length(states)), []),
  :pIdx => Latent(collect(1:length(phiVals)), [])
)
obs_L1 = Dict{Any,Observation}(
  :uIdx => Observation([:sIdx, :pIdx])
)

fg_l1 = compile_trace_to_factor_graph(tr_l1, lat_L1, obs_L1)
draw_factor_graph(fg_l1, graphviz, "L1", addr_to_name)