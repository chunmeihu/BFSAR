#!/usr/bin/env bash
# FashionIQ multi-view reranking. K = 200, lambda = (0.15, 0.40, 0.45).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "${PY}" -m bfsar.mvr.fuse_fiq \
  --k_rerank 200 \
  --k_save 200 \
  --fuse_mode frozen \
  --split val \
  --cqm_ckpt "${CQM:-${ROOT}/weights/cqm/fashioniq/cqm.pt}" \
  --stage1_ckpt "${CVS1:-${ROOT}/weights/cvs/fashioniq/stage1.pt}" \
  --stage2_ckpt "${CVS2:-${ROOT}/weights/cvs/fashioniq/stage2.pt}" \
  --init_model "${CKPT:-${ROOT}/weights/fgsa/fashioniq/pytorch_model.bin}" \
  --out_dir "${OUT:-${ROOT}/output/mvr/fashioniq}" \
  "$@"
