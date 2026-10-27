#!/usr/bin/env bash
# Composed query matching on CIRR. In-batch contrastive loss, temperature 0.07, 40 epochs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
NGPU="${NGPU:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
ARGS=(
  --stage a
  --exp_name "${EXP_NAME:-cqm_cirr}"
  --output_dir "${OUT:-${ROOT}/output/cqm/cirr}"
  --pretrained_id "${PRETRAINED_ID:-Salesforce/blip2-itm-vit-g}"
  --batch_size "${BATCH_SIZE:-16}"
  --blip_bs "${BLIP_BS:-8}"
  --grad_accumulation_step 1
  --global_contrastive
  --validate_every 2
  --epochs 40
  --blip_max_epoch 40
  --lr 2e-5
  --temperature 0.07
  --num_workers "${NUM_WORKERS:-2}"
  --k_val 50
)
if [[ "${NGPU}" -gt 1 ]]; then
  exec "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT:-29541}" \
    bfsar/mvr/train_cqm_cirr.py "${ARGS[@]}" "$@"
else
  exec "${PY}" bfsar/mvr/train_cqm_cirr.py "${ARGS[@]}" "$@"
fi
