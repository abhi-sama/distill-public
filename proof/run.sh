#!/usr/bin/env bash
# One stage of the end-to-end run at a time: proof/run.sh <decision-dir> <stage>
#
# Stages: synth, label, train, eval, export. The decision directory holds your own
# decision.yaml (see examples/decision.yaml); every stage writes its files beside it.
#
# DISTILL is a `distill` executable whose Python also has torch and laya (train, eval and
# export load models). One way to build it, from the repository root:
#   python3 -m venv venv-laya
#   venv-laya/bin/python -m pip install -e . torch transformers laya
#
# Everything runs locally through Ollama by default: the writer is `ollama-gemma-dense`
# (`ollama pull gemma4:31b-nvfp4`) and the labellers are `ollama-qwen` + `ollama-gemma`.
# Set WRITER to use another local model. A hosted API is never used unless you set REFERENCE
# (an evaluation comparison only) or FALLBACK (the exported routing snippet).
set -euo pipefail

dir=${1:?decision directory}
stage=${2:?stage}
ROOT=$(git rev-parse --show-toplevel)
DISTILL=${DISTILL:-distill}
DISTILL_PATH=$(command -v "$DISTILL")
case "$DISTILL_PATH" in
/*) ;;
*) DISTILL_PATH="$ROOT/${DISTILL_PATH#./}" ;;
esac
DISTILL=$DISTILL_PATH
BASE=${BASE:-convaiinnovations/laya}
DECISION=${DECISION:-decision.yaml}
SAMPLES_PER_STYLE=${SAMPLES_PER_STYLE:-60}
HELDOUT_COUNT=${HELDOUT_COUNT:-150}
LABEL_MAX_ITEMS=${LABEL_MAX_ITEMS:-}
WRITER=${WRITER:-ollama-gemma-dense}
cd "$dir"

case "$stage" in
synth)
    "$DISTILL" synth -d "$DECISION" --writer "$WRITER" \
        --samples-per-style "$SAMPLES_PER_STYLE" --heldout-count "$HELDOUT_COUNT"
    ;;
label)
    # Without a human reviewer, --teacher-gold writes the gold items both labellers agreed on,
    # and flagged training items keep their merged soft label.
    label_args=(-d "$DECISION" -i synthetic.jsonl -o labelled.jsonl \
        --queue review-queue.sqlite3 --gold-from heldout.jsonl --gold-size 75 \
        --gold-output gold.jsonl --heldout-output heldout-labelled.jsonl \
        --teacher-gold --include-unreviewed --batch-size 10)
    if [[ -n "$LABEL_MAX_ITEMS" ]]; then
        label_args+=(--max-items "$LABEL_MAX_ITEMS")
    fi
    "$DISTILL" label "${label_args[@]}"
    ;;
train)
    # TRAIN_RESUME=1 continues an interrupted run from its last saved epoch. TRAIN_ARGS adds
    # train options; the 5,000-state run passed --gradient-checkpointing to bound MPS memory.
    # shellcheck disable=SC2086
    "$DISTILL" train -d "$DECISION" --data labelled.jsonl -o model --base "$BASE" --device mps \
        ${TRAIN_RESUME:+--resume} ${TRAIN_ARGS:-}
    ;;
eval)
    # With no REFERENCE the readiness gate reports "no reference measured". REFERENCE can be
    # a local Ollama provider, or anthropic-api / openai-api (needs that API key).
    "$DISTILL" eval -d "$DECISION" -m model --heldout heldout-labelled.jsonl --gold gold.jsonl \
        -o eval-report --base "$BASE" --device mps ${REFERENCE:+--reference "$REFERENCE"}
    ;;
export)
    # TEACHERS records what labelled this directory; FALLBACK (default none) names the
    # teacher the routing snippet asks when the local model is not confident.
    "$DISTILL" export -d "$DECISION" -m model -e eval-report -o export --base "$BASE" \
        ${TEACHERS:---teacher ollama-qwen --teacher ollama-gemma} \
        ${FALLBACK:+--fallback "$FALLBACK"}
    ;;
*)
    echo "unknown stage $stage" >&2
    exit 2
    ;;
esac
