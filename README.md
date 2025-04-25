# Politeness RSA

Rational Speech Acts (RSA) is a model of replicating cognitive theories of human communication based on the concepts from Bayesian decision theory. RSA is a form of cognitive AI in linguistics, where we replicate various pragmatic phenomena through modeling the recursive reasoning process between speakers and listeners, such as how individuals infer implied meanings, resolve ambiguity, and understand context-dependent communication.

This repo contains an implementations of the politeness RSA model in Gen probailistic programming language. The politeness RSA model examines how polite language affects cooperative communication, where polite utterances may seem misleading but serve to protect the listener's and speaker's face. A detailed description of this model is available [here](https://www.problang.org/chapters/09-politeness.html). 

The implementation of this model was done in 3 alternative methods:

- Using VE‐based inference by [`GenVariableElimination.jl`](https://github.com/probcomp/GenVariableElimination.jl).
- Using Gen's new [`enumerative_inference`](https://www.gen.dev/docs/dev/ref/inference/enumerative/) method.
- Manual implementation of enumeration method.
