# Politeness RSA

Rational Speech Acts (RSA) is a Bayesian decision-theoretic framework for modeling how speakers and listeners recursively reason to infer implied meanings, resolve ambiguity, and handle context-dependent communication. This repository reproduces the full set of Politeness RSA results (the case study of white lies) in the [Gen](https://www.gen.dev/) probabilistic programming language and serves as a hands-on reference for those interested in RSA and Gen. 

The RSA of Politeness examines how polite language affects cooperative communication, where polite utterances may seem misleading but serve to protect the listener's and speaker's face. A detailed description of this model and its WebPPL implementation are available [here](https://www.problang.org/chapters/09-politeness.html).

The implementations are done in 3 alternative apporaches:

- VE‐based inference by [`GenVariableElimination.jl`](https://github.com/probcomp/GenVariableElimination.jl).
- Gen.jl's [`enumerative_inference`](https://www.gen.dev/docs/dev/ref/inference/enumerative/) method.
- Manual implementation.
