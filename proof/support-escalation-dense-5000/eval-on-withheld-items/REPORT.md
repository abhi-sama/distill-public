# Distill evaluation: NOT READY

Target selective accuracy: 97.0%

## Local

| Set | Accuracy | ECE (15) | Brier |
| --- | ---: | ---: | ---: |
| heldout | 90.67% | 0.0387 | 0.1560 |
| gold | 94.62% | 0.0328 | 0.0922 |

Latency: 16.4 ms/decision batched; 66.9 ms median for one state with all its questions; model load 2.4 s (excluded).
## Base Laya

| Set | Accuracy | ECE (15) | Brier |
| --- | ---: | ---: | ---: |
| heldout | 74.89% | 0.0488 | 0.3494 |
| gold | 75.38% | 0.0515 | 0.3431 |

Latency: 17.2 ms/decision batched; 67.9 ms median for one state with all its questions; model load 0.3 s (excluded).
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
| heldout | 75 | 97.33% | 0.0223 |
| gold | 65 | 96.92% | 0.0493 |


## Per question (gold)

| Question | n | Local | Base Laya | Claude |
| --- | ---: | ---: | ---: | ---: |
| escalate | 65 | 96.92% | 67.69% | n/a |
| legal_threat | 65 | 98.46% | 86.15% | n/a |
| safety_risk | 65 | 93.85% | 86.15% | n/a |
| security_incident | 65 | 90.77% | 63.08% | n/a |
| widespread_outage | 65 | 90.77% | 80.00% | n/a |
| manipulation | 65 | 96.92% | 69.23% | n/a |

## Cutoff

tau = 0.9676; local coverage = 54.22%; accepted accuracy = 97.13%.

## Claude baseline

Not measured: rerun with `DISTILL_LIVE_TEACHERS=1 --live-teachers` to enable the quota-aware Claude CLI baseline.

## Pass bar

**NOT READY**
- Claude gold baseline was not measured
