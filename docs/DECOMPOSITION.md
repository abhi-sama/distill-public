# Decomposition: fixed reconstruction rules

`distill/decomposition.py` turns the probabilities of several independent binary questions back
into the probability distribution of one original multi-option question. The rules below are
fixed before any labelling or scoring, so a decomposition experiment cannot be tuned to its own
evaluation labels. They are unit-tested in `tests/test_decomposition.py`.

The two worked examples are the mappings the module implements: a three-way moderation
`action`, and a PR review's `risk_level` and `primary_surface`. Each decomposed arm is scored
only after its binary probabilities pass through these rules, so it is compared with an
unchanged original-question control on the same states. This repository publishes the rules
only; it does not publish the experiment's states, labels or results.

## Content moderation

`action` is no longer a labelled three-way question. The model instead answers the existing
`harassment_or_hate`, `violent_threat`, and `spam_or_scam` questions plus two new binary
questions: `sexual_exploitation_or_self_harm` and `ambiguous_or_context_dependent`.

The fixed rule is:

1. `remove` if any of the four clear-violation flags is true.
2. Otherwise `review` if `ambiguous_or_context_dependent` is true.
3. Otherwise `keep`.

For calibration metrics, the corresponding probabilities are calculated in that same order:
the probability of removal is one minus the product of all clear-violation false
probabilities; review consumes the remaining mass times the ambiguity probability; keep is
the remaining mass. This is a declared independence composition, not a tuned classifier.

## PR security review

`risk_level` and `primary_surface` are no longer labelled multi-option questions. The model
answers binary surface questions for authentication, crypto/secrets, untrusted input, network,
dependencies, infrastructure/permissions, and personal data, plus three risk-condition
questions: high risk, security-control/exposed-surface change, and contained security-adjacent
change.

The fixed `risk_level` precedence is high (3), then control/exposed-surface (2), then contained
security-adjacent (1), then none (0). The fixed `primary_surface` precedence is untrusted
input, crypto/secrets, auth, network, infra/permissions, personal data, dependencies, then
none. A higher precedence applies whenever more than one binary surface is positive. Its
probability consumes the remaining probability mass before the next surface, so the eight
derived values form one calibrated distribution.

The precedence is intentionally safety-first and was not changed after seeing scores.
