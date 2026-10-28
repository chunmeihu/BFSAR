#!/usr/bin/env bash
# FashionIQ dual-path intent calibration.
# --split val prints recall. --split test has no local labels.
# Both paths use (alpha, beta) = (0.4, 0.2), w = 0.5.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

SPLIT="val"
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --split=*)
      SPLIT="${1#*=}"
      shift
      ;;
    *)
      PASS+=("$1")
      shift
      ;;
  esac
done

exec "${PY}" -m bfsar.dpic.eval_fiq \
  --split "${SPLIT}" \
  --ckpt "${CKPT:-${ROOT}/weights/fgsa/fashioniq/pytorch_model.bin}" \
  --out_dir "${OUT:-${ROOT}/output/dpic/fashioniq/${SPLIT}}" \
  --fgsa_levels 1 \
  "${PASS[@]}"
