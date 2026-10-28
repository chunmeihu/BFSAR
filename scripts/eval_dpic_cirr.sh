#!/usr/bin/env bash
# CIRR dual-path intent calibration.
# --split val prints recall. --split test is CIRR test1 and has no local labels.
# RASD (alpha, beta) = (0.5, 0.2), DASD (alpha, beta) = (0.2, 0.8), w = 0.2.
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

exec "${PY}" -m bfsar.dpic.eval_cirr \
  --split "${SPLIT}" \
  --ckpt "${CKPT:-${ROOT}/weights/fgsa/cirr/pytorch_model.bin}" \
  --out_dir "${OUT:-${ROOT}/output/dpic/cirr/${SPLIT}}" \
  --fgsa_levels 1 \
  "${PASS[@]}"
