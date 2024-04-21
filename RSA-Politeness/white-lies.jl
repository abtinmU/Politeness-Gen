
# Here, we model the case study of white lies, utterances which convey misleading information for purposes of politeness.

# Define state and utterance space
states = [1, 2, 3, 4, 5]
stateProbs = [0.2, 0.2, 0.2, 0.2, 0.2]
utterances = ["terrible", "bad", "okay", "good", "amazing"]
utterancesProbs = [0.2, 0.2, 0.2, 0.2, 0.2]
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

@dist meaning(utterance::String, state::Int64) =  bernoulli(prob(utterance, state))

####
#### Part 1: Literal Listener
####

@gen function literalListener(stateProbs::Vector{Float64}, utterance::String)

    state = @trace(Gen.categorical(stateProbs), :state)

    m = @trace(meaning(utterance, state), :m)

    return state

end

# The general distribution shae without observations (we expect uniform)
#
# traces =[Gen.simulate(literalListener, (stateProbs, "terrible")) for _=1:10000]
# state_posterior = [trace[:state] for trace in traces]
# 
# counts = countmap(state_posterior)
# 
# total_samples = length(state_posterior)
# proportions = Dict(state => count / total_samples for (state, count) in counts)
# 
# println(proportions)

# states = collect(keys(proportions))  # Get the keys for the x-axis
# state_proportions = collect(values(proportions))  # Get the values for the y-axis
# 
# bar(states, state_proportions, xlabel="States", ylabel="Proportions", legend=false)

# Using importance sampling for inference over observations
function L0_do_inference(model, stateProbs, utterance, m, amount_of_computation)

    observations = Gen.choicemap()
    observations[(:m)] = m
    (trace, _) = Gen.importance_resampling(model, (stateProbs, utterance), observations, amount_of_computation);

    return trace
end;

# traces = [L0_do_inference(literalListener, stateProbs, "good", true, 100) for _=1:1000]
# state_posterior = [trace[:state] for trace in traces]
# 
# counts = countmap(state_posterior)
# 
# total_samples = length(state_posterior)
# proportions = Dict(state => count / total_samples for (state, count) in counts)
# 
# println(proportions)
#
# states = collect(keys(proportions))  # Get the keys for the x-axis
# state_proportions = collect(values(proportions))  # Get the values for the y-axis
# 
# bar(states, state_proportions, xlabel="States", ylabel="Proportions", legend=false)

####
#### Part 2: Pragmatic Speaker
####

# We save the posterior distribiton of literal listener for easier access later on. 

function L0_posterior(utter::Int64)

    traces = [L0_do_inference(literalListener, stateProbs, utterances[utter], true, 500) for _=1:10000]
    state_posterior = [trace[:state] for trace in traces]
    counts = countmap(state_posterior)
    total_samples = length(state_posterior)
    proportions = Dict(state => count / total_samples for (state, count) in counts)

    return proportions
end

L0_posterior_map = Dict(
    "terrible" => L0_posterior(1),
    "bad" => L0_posterior(2),
    "okay" => L0_posterior(3),
    "good" => L0_posterior(4),
    "amazing" => L0_posterior(5),
)

L0_posterior_mapped(utter::Int64) = L0_posterior_map[utterances[utter]]
L0_posterior_mapped(utter::String) = L0_posterior_map[utter]

get_utterance(utterIndex::Int64) = utterances[utterIndex]

lambda_ = 1.25
social(proportions::Dict, valueFunctionLambda) = sum(key * value for (key, value) in proportions) * valueFunctionLambda

@dist L0_utterance(utterIndex::Int64) = categorical(L0_posterior_mapped(get_utterance(utterIndex)))

@dist drawUtterance(utterancesProbs::Vector{Float64}) = utterances[categorical(utterancesProbs)]

state_logProb(L0_post::Dict, state::Int64) = log(L0_post[state])

function S1_utility(state::Int64, utterance, phi::Float64)
    alpha_ = 10
    L0_post = L0_posterior_mapped(utterance)
    utility_epistemic = state_logProb(L0_post, state)
    utility_social = social(L0_post, lambda_)
    speakerUtility = phi * utility_epistemic + (1 - phi) * utility_social
    return alpha_ * speakerUtility
end

@gen function speaker1(state::Int64, phi::Float64)

    utterance = @trace(drawUtterance(utterancesProbs), :utterance)
    L0_post = L0_posterior_mapped(utterance)

    function S1_utility(state::Int64, utterance, phi::Float64)
        alpha_ = 10
        L0_post = L0_posterior_mapped(utterance)
        utility_epistemic = state_logProb(L0_post, state)
        utility_social = social(L0_post, lambda_)
        speakerUtility = phi * utility_epistemic + (1 - phi) * utility_social
        return alpha_ * speakerUtility
    end

    utility = @trace(normal(S1_utility(state, utterance, phi), 1), :utility)
    return [utility, utterance]

end

# traces =[Gen.simulate(speaker1, (4, 0.99)) for _=1:1000]
# trace_pl = [trace[:utterance] for trace in traces]
# counts = countmap(trace_pl)
# proportions = Dict(state => count / 1000 for (state, count) in counts)
# println(proportions)

# Save the untility values of prgmatic speaker for easy access during the inference
utils = Dict(state => (Dict(utterance => S1_utility(state, utterance, 0.99) for utterance in utterances)) for state in states)

@gen function S1_do_inference(model, state::Int64, phi::Float64, utils::Dict, amount_of_computation)

    observations = Gen.choicemap()
    observe_sorted = [[utter, utils[state][utter]] for utter in utterances]
    for i in observe_sorted
        observations[:utility] = i[2]
    end

    (trace, _) = Gen.importance_resampling(model, (state, phi), observations, amount_of_computation)
    return trace

end

# traces =[S1_do_inference(speaker1, 1, 0.99, utils, 100) for _=1:10000]
# trace_pl = [trace[:utterance] for trace in traces]
# counts = countmap(trace_pl)
# proportions = Dict(utt => count / 10000 for (utt, count) in counts)
# println(proportions)
#
# all_utters = Dict((utter, 0.0) for utter in utterances)
# 
# for (utter, prop) in proportions
#     if haskey(all_utters, utter)
#         all_utters[utter] = prop
#     end
# end
# 
# utter_proportions = [all_utters[utter] for utter in utterances]
# 
# bar(utterances, utter_proportions, xlabel="Utterances", ylabel="Proportions", legend=false)

####
#### Part 2: Pragmatic Listener
####

# First save the posterior of pragmatic speaker for easy accesss

function S1_posterior(state::Int64, phi::Float64)

    traces = [S1_do_inference(speaker1, state, phi, utils, 200) for _=1:5000]
    utter_posterior = [trace[:utterance] for trace in traces]
    counts = countmap(utter_posterior)
    total_samples = length(utter_posterior)
    incomplete_props = Dict(utter => count / total_samples for (utter, count) in counts)

    proportions = Dict((utter, 0.0) for utter in utterances)
    for (utter, prop) in incomplete_props
        if haskey(all_utters, utter)
            proportions[utter] = prop
        end
    end

    return proportions
end

phiVals = collect(0.05:0.05:0.95)
phiProbs = fill(1/length(phiVals), length(phiVals))

S1_posterior_map = Dict([state, phi] => S1_posterior(state, phi) for state in states for phi in phiVals)

S1_posterior_mapped(state::Int64, phi::Float64) = S1_posterior_map[[state, phi]]

@dist S1_utter_post(state::Int64, phi::Float64) = normal(S1_posterior_mapped(state, phi), 1)

@dist phi_(phiProbs::Vector) = phiVals[categorical(phiProbs)]

@gen function pragmaticListener(utterance)

    state = @trace(Gen.categorical(stateProbs), :state)

    phi = @trace(phi_(phiProbs), :phi)

    S1Dict = S1_posterior_mapped(state, phi)

    S1 = @trace(Gen.categorical(collect(values(S1Dict))), :S1)

    return [state, phi]
end

# traces =[Gen.simulate(pragmaticListener, ("good", )) for _=1:1000000]
# 
# state_posterior1 = [trace[:state] for trace in traces]
# state_posterior2 = [trace[:phi] for trace in traces]
# 
# counts1 = countmap(state_posterior1)
# counts2 = countmap(state_posterior2)
# 
# total_samples1 = length(state_posterior1)
# total_samples2 = length(state_posterior2)
# 
# proportions1 = Dict(state => count / total_samples1 for (state, count) in counts1)
# proportions2 = Dict(state => count / total_samples2 for (state, count) in counts2)
# 
# println("state posterior: ", proportions1)
# println("phi posterior: ", proportions2)

function L1_do_inference(model, utterance, amount_of_computation)

    utter_map = Dict("okay"=>1, "amazing"=>2, "terrible"=>3, "bad"=>4, "good"=>5)
    observations = Gen.choicemap()
    observations[(:S1)] = utter_map[utterance]

    (trace, _) = Gen.importance_resampling(model, (utterance, ), observations, amount_of_computation);
    return trace

end;

traces =[L1_do_inference(pragmaticListener, "bad", 100) for _=1:100000]

state_posterior1 = [trace[:state] for trace in traces]
state_posterior2 = [trace[:phi] for trace in traces]

counts1 = countmap(state_posterior1)
counts2 = countmap(state_posterior2)

total_samples1 = length(state_posterior1)
total_samples2 = length(state_posterior2)

proportions1 = Dict(state => count / total_samples1 for (state, count) in counts1)
proportions2 = Dict(state => count / total_samples2 for (state, count) in counts2)

println("state posterior: ", proportions1)
println("phi posterior: ", proportions2)

states1 = collect(keys(proportions1))
state_proportions1 = collect(values(proportions1))
bar(states1, state_proportions1, xlabel="States", ylabel="State Proportions", legend=false)

phi_num = collect(keys(proportions2))
phi_proportions = collect(values(proportions2)) 
bar(phi_num, phi_proportions, xlabel="Phi Values", ylabel="Phi Proportions", legend=false)