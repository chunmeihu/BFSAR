#!/usr/bin/env bash
# Candidate verification scoring.
# The numbers in the paper use the released stage-2 checkpoints under weights/cvs/.
# This entry retrains that head. Run from the CVS package so its local imports resolve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
DATASET="${DATASET:-CIRR}"
if [[ "${DATASET}" == "CIRR" ]]; then
  STAGE1="${STAGE1:-${ROOT}/weights/cvs/cirr/stage1.pt}"
else
  STAGE1="${STAGE1:-${ROOT}/weights/cvs/fashioniq/stage1.pt}"
fi
cd "${ROOT}/bfsar/mvr/cvs"
export PYTHONPATH="${ROOT}/bfsar/mvr/cvs/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PY}" src/stage2_train.py \
  --train \
  --dataset "${DATASET}" \
  --top-k-path "${TOPK:?set TOPK to a top-200 candidate file}" \
  --K-value "${K:-200}" \
  --blip-model-path "${STAGE1}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --blip-learning-rate 2e-5 \
  --num-epochs "${EPOCHS:-20}" \
  "$@"
