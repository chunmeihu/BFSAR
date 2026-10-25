#!/usr/bin/env bash
# QTR Stage-A: train Composer (Q-Former_C + projections) with CIR ITC; freeze later in Stage-B.
# Independent experiment tree: output/cirr_rerank/blip2_qtr/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
export VIRTUAL_ENV="${ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export ASSET_ROOT="${ASSET_ROOT:-$HOME}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Isolate from other CIRR jobs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29531}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"   # Stage-A ITC is cheap → larger global CE
BLIP_BS="${BLIP_BS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
PRETRAINED_ID="${PRETRAINED_ID:-Salesforce/blip2-itm-vit-g}"
GLOBAL_B=$((BATCH_SIZE * NGPU))
EXP_NAME="${EXP_NAME:-qtr_v1_stageA_g${GLOBAL_B}}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/cirr_rerank/blip2_qtr}"
EPOCHS="${EPOCHS:-40}"
VAL_EVERY="${VAL_EVERY:-2}"
LR="${LR:-2e-5}"

cd "${ROOT}"
# Refuse launch if GPUs are busy (>2GB used)
python - <<'PY'
import subprocess, sys
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
busy = []
for line in out.strip().splitlines():
    idx, used = [x.strip() for x in line.split(",")]
    if float(used) > 2048:
        busy.append(f"GPU{idx}={used}MiB")
if busy:
    print("ABORT: GPUs not idle:", ", ".join(busy), file=sys.stderr)
    sys.exit(2)
print("GPU idle check OK")
PY

ARGS=(
  --stage a
  --exp_name "${EXP_NAME}"
  --pretrained_id "${PRETRAINED_ID}"
  --batch_size "${BATCH_SIZE}"
  --blip_bs "${BLIP_BS}"
  --grad_accumulation_step "${GRAD_ACCUM}"
  --global_contrastive
  --validate_every "${VAL_EVERY}"
  --epochs "${EPOCHS}"
  --blip_max_epoch "${EPOCHS}"
  --lr "${LR}"
  --temperature 0.07
  --num_workers "${NUM_WORKERS}"
  --output_dir "${OUT_ROOT}"
  --local_files_only
  --k_val 50
  "$@"
)

echo "============================================================"
echo "QTR Stage-A | exp=${EXP_NAME}"
echo "Backbone=${PRETRAINED_ID} | local_bs=${BATCH_SIZE} ×${NGPU} => global_CE=${GLOBAL_B}"
echo "Out: ${OUT_ROOT}/${EXP_NAME}"
echo "Env: ${VIRTUAL_ENV}"
echo "============================================================"

if [[ "$NGPU" -gt 1 ]]; then
  exec "${VENV_PY}" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    HBI/cirr/vision_desc/train_cirr_blip2_qtr.py "${ARGS[@]}"
else
  exec "${VENV_PY}" HBI/cirr/vision_desc/train_cirr_blip2_qtr.py "${ARGS[@]}"
fi
