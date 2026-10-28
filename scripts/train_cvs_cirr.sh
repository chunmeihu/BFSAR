#!/usr/bin/env bash
# CIRR candidate verification: stage 1, top-200 lists, then stage 2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
cd "${ROOT}/bfsar/mvr/cvs"
"${PY}" src/stage1_train.py --dataset cirr \
  --num-epochs 40 \
  --blip-max-epoch 10 \
  --blip-learning-rate 2e-5 \
  --batch-size 512 \
  --blip-bs 16 \
  --num_workers 4 \
  --transform targetpad --target-ratio 1.25 \
  --validation-frequency 1 \
  --save-training --save-best
"${PY}" src/validate.py --dataset cirr --k 200
"${PY}" src/stage2_train.py --dataset cirr \
  --num-epochs 100 \
  --blip-max-epoch 80 \
  --blip-learning-rate 2e-5 \
  --batch-size 16 \
  --blip-bs 16 \
  --K-value 50 \
  --num_workers 4 \
  --transform targetpad --target-ratio 1.25 \
  --validation-frequency 1 \
  --save-training --save-best
