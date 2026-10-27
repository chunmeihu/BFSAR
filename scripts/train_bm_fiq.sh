#!/usr/bin/env bash
# Banzhaf teacher on FashionIQ. CLIP is frozen. 50 Monte Carlo samples, 10 epochs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
NGPU="${NGPU:-2}"
OUT="${OUT:-${ROOT}/output/bm/fashioniq}"
CLIP="${CLIP:-${ROOT}/weights/openclip/ViT-L-14-laion2B-s32B-b82K.pt}"
mkdir -p "${OUT}"
ARGS=(
  --output_dir "${OUT}"
  --batch_size "${BATCH_SIZE:-32}"
  --grad_accum_steps "${GRAD_ACCUM:-4}"
  --epochs 10
  --lr 1e-2
  --workers "${NUM_WORKERS:-4}"
  --max_frames 12
  --max_words 32
  --banzhaf_mc_samples 50
  --clip_checkpoint "${CLIP}"
)
if [[ "${NGPU}" -gt 1 ]]; then
  exec "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT:-29522}" \
    train_bm_fiq.py "${ARGS[@]}" "$@"
else
  exec "${PY}" train_bm_fiq.py "${ARGS[@]}" "$@"
fi
