#!/usr/bin/env bash
# FashionIQ dual-path intent calibration.
# Both paths use (alpha, beta) = (0.4, 0.2), w = 0.5.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "${PY}" -m bfsar.dpic.eval_fiq \
  --ckpt "${CKPT:-${ROOT}/weights/fgsa/fashioniq/pytorch_model.bin}" \
  --out_dir "${OUT:-${ROOT}/output/dpic/fashioniq}" \
  --fgsa_levels 1 \
  "$@"
