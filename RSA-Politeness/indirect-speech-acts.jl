
# Here we model politeness with indirect speech acts. Where speakers may deliberately be indirect for considerations of politeness.

utterances = [
  "yes_terrible","yes_bad","yes_good","yes_amazing",
  "not_terrible","not_bad","not_good","not_amazing"
]
utterancesProbs = [1/8, 1/8, 1/8, 1/8,
                   1/8, 1/8, 1/8, 1/8]
states = [0,1,2,3]
stateProbs = [0.25, 0.25, 0.25, 0.25]

function isNegation(utt)
    return (split(utt, "_")[1] == "not")
end

cost_yes = 0;
cost_neg = 0.35
speakerOptimality = 4
speakerOptimality2 = 4

function roundToTwo(x)
    return round(x, digits=2)
end

weightBins = round.(range(0, stop=1, step=0.05), digits=2)

phiWeights = repeat([1/length(weightBins)], length(weightBins))

function cost(utterance)
    return isNegation(utterance) ? cost_neg : cost_yes
end

literalSemantics = Dict(
  "state"=> [0, 1, 2, 3],
  "not_amazing"=> [0.9652,0.9857,0.7873,0.0018],
  "not_bad"=> [0.0967,0.365,0.7597,0.9174],
  "not_good"=> [0.9909,0.736,0.2552,0.2228],
  "not_terrible"=> [0.2749,0.5285,0.728,0.9203],
  "yes_amazing"=> [4e-04,2e-04,0.1048,0.9788 ],
  "yes_bad"=> [0.9999,0.8777,0.1759,0.005],
  "yes_good"=> [0.0145,0.1126,0.9893,0.9999],
  "yes_terrible"=> [0.9999,0.3142,0.0708,0.0198]
)

prob(utterance::String, state::Int64) = literalSemantics[utterance][state]

@dist meaning(utterance::String, state::Int64) =  bernoulli(prob(utterance, state))

####
#### Part 1: Literal Listener
####

@gen function literalListener(stateProbs::Vector{Float64}, utterance::String)

    state = @trace(state_sample(stateProbs), :state) + 1

    m = @trace(meaning(utterance, state), :m)

    return state

end

function L0_do_inference(model, stateProbs, utterance, m, amount_of_computation)

    observations = Gen.choicemap()
    observations[(:m)] = m
    (trace, _) = Gen.importance_resampling(model, (stateProbs, utterance), observations, amount_of_computation);

    return trace
end;

# traces = [L0_do_inference(literalListener, stateProbs, "not_amazing", true, 100) for _=1:10000]
# state_posterior = [trace[:state] for trace in traces]
# 
# counts = countmap(state_posterior)
# 
# total_samples = length(state_posterior)
# proportions = Dict(state => count / total_samples for (state, count) in counts)
# 
# println(proportions)
# 
# all_states = Dict((state, 0.0) for state in states)
# 
# for (state, prop) in proportions
#     if haskey(all_states, state)
#         all_states[state] = prop
#     end
# end
# 
# state_proportions = [all_states[state] for state in states]
# 
# bar(states, state_proportions, xlabel="States", ylabel="Proportions", legend=false)

####
#### Part 2: Literal Listener
####

# Save the posterior of literal listener for easy (& fast) access during pragmatic speaker's inference

function L0_posterior(utter::Int64)

    traces = [L0_do_inference(literalListener, stateProbs, utterances[utter], true, 500) for _=1:10000]
    state_posterior = [trace[:state] for trace in traces]
    counts = countmap(state_posterior)
    total_samples = length(state_posterior)
    proportions = Dict(state => count / total_samples for (state, count) in counts)

    return proportions
end

L0_posterior_map = Dict(
  "yes_terrible"=> L0_posterior(1),
  "yes_bad"=> L0_posterior(2),
  "yes_good"=> L0_posterior(3),
  "yes_amazing"=> L0_posterior(4),
  "not_terrible"=> L0_posterior(5),
  "not_bad"=> L0_posterior(6),
  "not_good"=> L0_posterior(7),
  "not_amazing"=> L0_posterior(8)
)

L0_posterior_mapped(utter::Int64) = L0_posterior_map[utterances[utter]]
L0_posterior_mapped(utter::String) = L0_posterior_map[utter]

get_utterance(utterIndex::Int64) = utterances[utterIndex]

lambda_ = 1.25
social(proportions::Dict, valueFunctionLambda) = sum(key * value for (key, value) in proportions) * valueFunctionLambda

@dist L0_utterance(utterIndex::Int64) = categorical(L0_posterior_mapped(get_utterance(utterIndex)))
@dist drawUtterance(utterancesProbs::Vector{Float64}) = categorical(utterancesProbs)

state_logProb(L0_post::Dict, state::Int64) = log(L0_post[state])

L0_stateFunc(utterance::Int64) = [0, L0_posterior_mapped(utterance)]

@dist L0_state(utterance::Int64) = L0_stateFunc(utterance)[categorical([0, 1])]

@gen function speaker1(state::Int64, phi::Float64)

    utterance = @trace(drawUtterance(utterancesProbs), :utterance)
    L0_post = L0_posterior_mapped(utterance)

    function S1_utility(state::Int64, utterance, phi::Float64)
        L0_post = L0_posterior_mapped(utterance)
        utility_epistemic = state_logProb(L0_post, state)
        utility_social = social(L0_post, lambda_)
        speakerUtility = phi * utility_epistemic + (1 - phi) * utility_social - cost(utterances[utterance])
        return speakerOptimality * speakerUtility
    end

    utility = @trace(normal(S1_utility(state, utterance, phi), 1), :utility)
    return [utility, utterance]

end

# Define the utility function of the pragmatic speaker outside the speaker1 model. This would allow us to save the utility values of 
#speaker1 in a dictionary for easy later access.

function S1_utility(state::Int64, utterance, phi::Float64)
    L0_post = L0_posterior_mapped(utterance)
    utility_epistemic = state_logProb(L0_post, state)
    utility_social = social(L0_post, lambda_)
    speakerUtility = phi * utility_epistemic + (1 - phi) * utility_social - cost(utterance)
    return speakerOptimality * speakerUtility
end

# Now save the utility values
utils = Dict(state => (Dict(utterance => S1_utility(state, utterance, 0.99) for utterance in utterances)) for state in states)

@gen function S1_do_inference3(model, state::Int64, phi::Float64, utils::Dict, amount_of_computation)

    observations = Gen.choicemap()
    observe_sorted = [[utter, utils[state][utter]] for utter in utterances]

    for i in observe_sorted
        observations[:utility] = i[2]
    end

    (trace, _) = Gen.importance_resampling(model, (state, phi), observations, amount_of_computation)
    return trace
end

# traces =[S1_do_inference3(speaker1, 1, 0.99, utils, 100) for _=1:10000]
# 
# trace_pl = [trace[:utterance] for trace in traces]
# 
# counts = countmap(trace_pl)
# 
# proportions = Dict(utt => count / 10000 for (utt, count) in counts)
# 
# println(proportions)
# all_utters = Dict((utter, 0.0) for utter in utterances)
# 
# for (utter, prop) in proportions
#     all_utters[utterances[utter]] = prop
# end
# 
# utter_proportions = [all_utters[utter] for utter in utterances]
# 
# bar(utterances, utter_proportions, xlabel="Utterances", ylabel="Proportions", legend=false)

####
#### Part 3: Literal Listener
####

# Save the posterior of speaker1 for easy later access

function S1_posterior(state::Int64, phi::Float64)

    traces = [S1_do_inference3(speaker1, state, phi, utils, 200) for _=1:5000]
    utter_posterior = [trace[:utterance] for trace in traces]
    counts = countmap(utter_posterior)
    total_samples = length(utter_posterior)
    incomplete_props = Dict(utter => count / total_samples for (utter, count) in counts)

    proportions = Dict((utter, 0.0) for utter in utterances)
    for (utter, prop) in incomplete_props
        proportions[utterances[utter]] = prop
    end

    return proportions
end

S1_posterior_map = Dict([state, phi] => S1_posterior(state, phi) for state in states for phi in weightBins)

S1_posterior_mapped(state::Int64, phi::Float64) = S1_posterior_map[[state, phi]]

@dist phi_(phiWeights::Vector) = weightBins[categorical(phiWeights)]

@gen function pragmaticListener(utterance)

    state = @trace(state_sample(stateProbs), :state) + 1
    phi = @trace(phi_(phiWeights), :phi)
    S1Dict = S1_posterior_mapped(state-1, phi)

    S1 = @trace(Gen.categorical(collect(values(S1Dict))), :S1)
    return [state, phi]

end

function L1_do_inference(model, utterance, amount_of_computation)

    utter_map = Dict("not_bad"=> 1,
                     "yes_amazing"=>2,
                     "yes_terrible"=>3,
                     "not_good"=>4,
                     "yes_good"=>5,
                     "yes_bad"=>6,
                     "not_terrible"=>7,
                     "not_amazing"=>8)

    observations = Gen.choicemap()
    observations[(:S1)] = utter_map[utterance]
    (trace, _) = Gen.importance_resampling(model, (utterance, ), observations, amount_of_computation);

    return trace
end

# traces =[L1_do_inference(pragmaticListener, "not_bad", 100) for _=1:100000]
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
# 
# states1 = collect(keys(proportions1))  # Get the keys for the x-axis
# state_proportions1 = collect(values(proportions1))  # Get the values for the y-axis
# 
# bar(states1, state_proportions1, xlabel="States", ylabel="State Proportions", legend=false)
# 
# phi_num = collect(keys(proportions2))  # Get the keys for the x-axis
# phi_proportions = collect(values(proportions2))  # Get the values for the y-axis
# 
# bar(phi_num, phi_proportions, xlabel="Phi Values", ylabel="Phi Proportions", legend=false)

####
#### Part 4: Literal Listener
####

# Again, I save the posterior of the previous model for easy (&fast) later access

function L1_posterior(utter::Int64)

    traces = [L1_do_inference(pragmaticListener, utterances[utter], 100) for _=1:10000]

    state_posterior = [trace[:state] for trace in traces]
    phi_posterior = [trace[:phi] for trace in traces]

    countState = countmap(state_posterior)
    countPhi = countmap(phi_posterior)

    total_samples_state = length(state_posterior)
    total_samples_phi = length(phi_posterior)

    state_incomplete_props = Dict(state => count / total_samples_state for (state, count) in countState)
    phi_incomplete_props = Dict(state => count / total_samples_phi for (state, count) in countPhi)

    proportions_state = Dict((state, 0.0) for state in states)
    for (state, prop) in state_incomplete_props
        proportions_state[state] = prop
    end

    proportions_phi = Dict((phi, 0.0) for phi in weightBins)
    for (phi, prop) in phi_incomplete_props
        proportions_phi[phi] = prop
    end

    return [proportions_state, proportions_phi]
end

s1, p1 = L1_posterior(1); s2, p2 = L1_posterior(2); s3, p3 = L1_posterior(3); s4, p4 = L1_posterior(4)
s5, p5 = L1_posterior(5); s6, p6 = L1_posterior(6); s7, p7 = L1_posterior(7); s8, p8 = L1_posterior(8)

L1_posterior_map_state = Dict("yes_terrible"=>s1,"yes_bad"=>s2,"yes_good"=>s3,"yes_amazing"=>s4,
                              "not_terrible"=>s5,"not_bad"=>s6,"not_good"=>s7,"not_amazing"=> s8)
L1_posterior_map_phi = Dict("yes_terrible"=>p1,"yes_bad"=>p2,"yes_good"=>p3,"yes_amazing"=>p4,
                            "not_terrible"=>p5,"not_bad"=>p6,"not_good"=>p7,"not_amazing"=>p8)

L1_posterior_mapped_state(utter::Int64) = L1_posterior_map_state[utterances[utter]]
L1_posterior_mapped_state(utter::String) = L1_posterior_map_state[utter]
L1_posterior_mapped_phi(utter::Int64) = L1_posterior_map_phi[utterances[utter]]
L1_posterior_mapped_phi(utter::String) = L1_posterior_map_phi[utter]

get_utterance(utterIndex::Int64) = utterances[utterIndex]

lambda_ = 1.25
social(proportions::Dict, valueFunctionLambda) = sum(key * value for (key, value) in proportions) * valueFunctionLambda

@dist L0_utterance_state(utterIndex::Int64) = categorical(L1_posterior_mapped_state(utterIndex))
@dist L0_utterance_phi(utterIndex::Int64) = categorical(L1_posterior_mapped_phi(utterIndex))
@dist drawUtterance(utterancesProbs::Vector{Float64}) = categorical(utterancesProbs)

@gen function speaker2(state::Int64, phi::Float64, weights::Dict)

    utterance = @trace(drawUtterance(utterancesProbs), :utterance)

    function S2_utility(state::Int64, utterance, phi::Float64, weights::Dict)
        L1_state = L1_posterior_mapped_state(utterance)
        L1_phi = L1_posterior_mapped_phi(utterance)
        utilities_epistemic = state_logProb(L1_state, state)
        utilities_social = social(L1_state, lambda_)
        utilities_presentational = social(L1_phi, lambda_)
        speakerUtility = weights["inf"] * utilities_epistemic + weights["soc"] * utilities_social + weights["pres"] * utilities_presentational
        return speakerOptimality2 * speakerUtility
    end

    utility = @trace(normal(S2_utility(state, utterance, phi, weights), 1), :utility)
    return utterances[utterance]

end

@gen function S2_do_inference(model, state::Int64, phi::Float64, weights::Dict, amount_of_computation)

    function S2_utility(state::Int64, utterance, phi::Float64, weights::Dict)
        L1_state = L1_posterior_mapped_state(utterance)
        L1_phi = L1_posterior_mapped_phi(utterance)
        utilities_epistemic = state_logProb(L1_state, state)
        utilities_social = social(L1_state, lambda_)
        utilities_presentational = social(L1_phi, lambda_)
        speakerUtility = weights["inf"] * utilities_epistemic + weights["soc"] * utilities_social + weights["pres"] * utilities_presentational
        return speakerOptimality2 * speakerUtility
    end

    observations = Gen.choicemap()
    observe_sorted = [[utter, S2_utility(state, utter, phi, weights)] for utter in utterances]
    # state_map = Dict(0 => 1, 2 => 2, 3 => 3, 1 => 4)
    # phi_map = Dict(0.8=>1, 0.95=>2, 0.3=>3, 0.5=>4, 0.55=>5, 0.1=>6, 0.45=>7, 0.25=>8, 0.35=>9, 1.0=>10,
    #                0.7=>11, 0.0=>12, 0.4=>13, 0.85=>14, 0.15=>15,0.2=>16, 0.9=>17, 0.65=>18, 0.75=>19, 0.05=>20, 0.6=>21)
    for i in observe_sorted
        observations[(:utility)] = i[2]
    end

    (trace, _) = Gen.importance_resampling(model, (state, phi, weights), observations, amount_of_computation)
    return trace

end

traces =[S2_do_inference(speaker2, 0, 0.35, Dict("soc"=> 0.30, "pres"=> 0.45, "inf"=> 0.25), 100) for _=1:10000]
trace_pl = [trace[:utterance] for trace in traces]
counts = countmap(trace_pl)
proportions = Dict(utt => count / 10000 for (utt, count) in counts)
println(proportions)

all_utters = Dict((utter, 0.0) for utter in utterances)
for (utter, prop) in proportions
    all_utters[utterances[utter]] = prop
end

utter_proportions = [all_utters[utter] for utter in utterances]
bar(utterances, utter_proportions, xlabel="Utterances", ylabel="Proportions", legend=false)
