#!/usr/bin/env bash
# CIRR A3t skl=0.1: exclusive 2×16 train, then val MIC + dual (for SKL pick).
# test1 submissions are queued only if this SKL wins vs A2e.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
export VIRTUAL_ENV="${ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER 2>/dev/null || true

LOGDIR="${ROOT}/output/ablations/logs"
mkdir -p "${LOGDIR}"
cd "${ROOT}"

TRAIN="${ROOT}/output/ablations/cirr_coarse/A3t_skl01_entity"
DUAL="${ROOT}/output/ablations/cirr_coarse/A3t_skl01_entity_dual"

echo "[A3t skl=0.1] started=$(date -Iseconds)"

used0="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{print int($1)}')"
used1="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | awk '{print int($1)}')"
if [[ "${used0}" -gt 500 || "${used1}" -gt 500 ]]; then
  echo "ABORT: GPUs not empty (used0=${used0} used1=${used1} MiB). Not stealing cards." >&2
  exit 3
fi

if compgen -G "${TRAIN}/pytorch_model.bin.best.*" > /dev/null; then
  echo "=== skip A3t train (best ckpt already exists) ==="
else
  echo "=== CIRR A3t train (Banzhaf entity-only, skl=0.1, 2×16 DDP) ==="
  export CUDA_VISIBLE_DEVICES=0,1
  export NGPU=2
  export BATCH_SIZE=16
  export BATCH_SIZE_VAL=16
  bash "${ROOT}/scripts/ablations/run_cirr_coarse_train_A3_skl01.sh"
fi

unset CUDA_VISIBLE_DEVICES || true

if [[ -f "${TRAIN}/metrics.json" ]]; then
  echo "=== skip A3t MIC (already has metrics.json) ==="
else
  echo "=== CIRR A3t val MIC ==="
  export CUDA_VISIBLE_DEVICES=0
  "${VENV_PY}" -u -m ablations.coarse_retrieval.eval_a2_a3_mic_wti \
    --dataset cirr --ckpt "${TRAIN}" --out_dir "${TRAIN}" \
    --wti_levels 1 --device cuda:0
  unset CUDA_VISIBLE_DEVICES || true
fi

if [[ -f "${DUAL}/metrics.json" ]]; then
  echo "=== skip A3t val dual (already has metrics.json) ==="
else
  echo "=== CIRR A3t val dual (frozen A7 α/β/w) ==="
  export CUDA_VISIBLE_DEVICES=0
  "${VENV_PY}" -u -m ablations.coarse_retrieval.eval_a3_dual_cirr \
    --ckpt "${TRAIN}" \
    --out_dir "${DUAL}" \
    --wti_levels 1 \
    --device cuda:0 \
    --memory_fraction 0.90 \
    --exp_id A3t_dual \
    --exp_name entity_skl01_dual
  unset CUDA_VISIBLE_DEVICES || true
fi

"${VENV_PY}" -u -m ablations.coarse_retrieval.report_coarse_table
"${VENV_PY}" -u -m ablations.coarse_retrieval.pick_cirr_banzhaf_vs_a2e --require-a3t
date -Iseconds > "${TRAIN}/train_done.txt"
echo "OK: CIRR A3t skl=0.1 train+val MIC/dual done." | tee "${LOGDIR}/cirr_a3t_skl01_done.txt"
echo "finished=$(date -Iseconds)"
