#!/usr/bin/env bash
# Single-level FGSA on CIRR. Loss is L_CON + lambda_B * L_KL, lambda_B = 0.1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
NGPU="${NGPU:-2}"
OUT="${OUT:-${ROOT}/output/fgsa/cirr}"
CLIP="${CLIP:-${ROOT}/weights/openclip/ViT-L-14-laion2B-s32B-b82K.pt}"
TEACHER="${TEACHER:-${ROOT}/weights/bm/cirr/banzhaf_module.pt}"
mkdir -p "${OUT}"
ARGS=(
  --do_train 1
  --output_dir "${OUT}"
  --batch_size "${BATCH_SIZE:-16}"
  --batch_size_val "${BATCH_SIZE_VAL:-16}"
  --epochs 10
  --lr 1e-4
  --warmup_proportion 0.1
  --workers "${NUM_WORKERS:-4}"
  --estimator "${TEACHER}"
  --lambda_b 0.1
  --fgsa_levels 1
  --max_frames 12
  --max_words 32
  --v_rate0 0.25
  --v_rate1 0.5
  --image_encode_microbatch 12
  --clip_checkpoint "${CLIP}"
)
if [[ "${NGPU}" -gt 1 ]]; then
  exec "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT:-29531}" \
    train_fgsa_cirr_openclip.py "${ARGS[@]}" "$@"
else
  exec "${PY}" train_fgsa_cirr_openclip.py "${ARGS[@]}" "$@"
fi
