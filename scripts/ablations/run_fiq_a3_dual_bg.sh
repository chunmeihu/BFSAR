#!/usr/bin/env bash
# FIQ A3 entity-only Banzhaf + dual-path coarse, sharing leftover GPU memory
# with the running CIRR A2 train. Does not touch A2 PIDs / mainline ckpts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
LOGDIR="${ROOT}/output/ablations/logs"
OUT="${ROOT}/output/ablations/fiq_coarse/A3_entity_only_dual"
CKPT="${ROOT}/output/ablations/fiq_coarse/A3_entity_only/pytorch_model.bin.best.4"
mkdir -p "${LOGDIR}" "${OUT}"
cd "${ROOT}"

export PYTHONPATH="${ROOT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Physical GPU 1 (A2 rank1 already ~9GB). Cap this job inside the python script.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "FIQ A3 dual coarse $(date -Iseconds)"
echo "ckpt=${CKPT}"
echo "out=${OUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

exec "${VENV_PY}" -u -m ablations.coarse_retrieval.eval_a3_dual_fiq \
  --ckpt "${CKPT}" \
  --out_dir "${OUT}" \
  --device cuda:0 \
  --wti_levels 1 \
  --memory_fraction 0.40 \
  --batch_size_val 8 \
  --query_batch 16 \
  --gallery_batch 32 \
  --workers 2 \
  "$@"
