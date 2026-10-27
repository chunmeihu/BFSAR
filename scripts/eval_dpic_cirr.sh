#!/usr/bin/env bash
# CIRR dual-path intent calibration.
# RASD (alpha, beta) = (0.5, 0.2), DASD (alpha, beta) = (0.2, 0.8), w = 0.2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "${PY}" -m bfsar.dpic.eval_cirr \
  --ckpt "${CKPT:-${ROOT}/weights/fgsa/cirr/pytorch_model.bin}" \
  --out_dir "${OUT:-${ROOT}/output/dpic/cirr}" \
  --fgsa_levels 1 \
  "$@"
