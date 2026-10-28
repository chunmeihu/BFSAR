#!/usr/bin/env bash
# FashionIQ candidate verification: stage 1, top-200 lists, then stage 2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
cd "${ROOT}/bfsar/mvr/cvs"
"${PY}" src/stage1_train.py --dataset fashioniq \
  --num-epochs 20 \
  --blip-max-epoch 20 \
  --blip-learning-rate 2e-5 \
  --batch-size 512 \
  --blip-bs 16 \
  --num_workers 2 \
  --transform targetpad --target-ratio 1.25 \
  --validation-frequency 1 \
  --save-training --save-best
"${PY}" src/validate.py --dataset fashioniq --k 200
"${PY}" src/stage2_train.py --dataset fashioniq \
  --num-epochs 50 \
  --blip-max-epoch 50 \
  --blip-learning-rate 2e-5 \
  --batch-size 16 \
  --blip-bs 16 \
  --K-value 100 \
  --num_workers 2 \
  --transform targetpad --target-ratio 1.25 \
  --validation-frequency 1 \
  --save-training --save-best
