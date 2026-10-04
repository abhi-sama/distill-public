# Distill evaluation: NOT READY

Target selective accuracy: 97.0%

## Local

| Set | Accuracy | ECE (15) | Brier |
| --- | ---: | ---: | ---: |
| heldout | 99.11% | 0.0337 | 0.0238 |
| gold | 98.04% | 0.0340 | 0.0337 |

Latency: 12.1 ms/decision batched; 63.6 ms median for one state with all its questions; model load 3.1 s (excluded).
## Base Laya

| Set | Accuracy | ECE (15) | Brier |
| --- | ---: | ---: | ---: |
| heldout | 76.89% | 0.0558 | 0.3388 |
| gold | 75.74% | 0.0540 | 0.3466 |

Latency: 12.2 ms/decision batched; 67.8 ms median for one state with all its questions; model load 1.0 s (excluded).
Marginal external cost: $0.0000/1k decisions.

## Adversarial slices

### prompt injection

| Set | n | Accuracy | ECE (15) |
| --- | ---: | ---: | ---: |
| heldout | 0 | n/a | n/a |
| gold | 0 | n/a | n/a |

### manipulation question

| Set | n | Accuracy | ECE (15) |
| --- | ---: | ---: | ---: |
| heldout | 75 | 100.00% | 0.0175 |
| gold | 68 | 100.00% | 0.0179 |


## Per question (gold)

| Question | n | Local | Base Laya | Claude |
| --- | ---: | ---: | ---: | ---: |
| escalate | 68 | 95.59% | 64.71% | n/a |
| legal_threat | 68 | 98.53% | 92.65% | n/a |
| safety_risk | 68 | 100.00% | 98.53% | n/a |
| security_incident | 68 | 95.59% | 60.29% | n/a |
| widespread_outage | 68 | 98.53% | 72.06% | n/a |
| manipulation | 68 | 100.00% | 66.18% | n/a |

## Cutoff

tau = 0.0000; local coverage = 100.00%; accepted accuracy = 99.11%.

## Claude baseline

Not measured: rerun with `DISTILL_LIVE_TEACHERS=1 --live-teachers` to enable the quota-aware Claude CLI baseline.

## Pass bar

**NOT READY**
- Claude gold baseline was not measured
