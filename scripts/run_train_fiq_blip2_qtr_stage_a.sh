#!/usr/bin/env bash
# Fashion-IQ QTR Stage-A: train Composer (Q-Former_C + projections) with CIR ITC.
# Mixed dress+shirt+toptee. Does not load CIRR weights. Writes under output/fiq_rerank/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
export VIRTUAL_ENV="${ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export ASSET_ROOT="${ASSET_ROOT:-$HOME}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29541}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BLIP_BS="${BLIP_BS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
PRETRAINED_ID="${PRETRAINED_ID:-Salesforce/blip2-itm-vit-g}"
GLOBAL_B=$((BATCH_SIZE * NGPU))
EXP_NAME="${EXP_NAME:-qtr_v1_stageA_g${GLOBAL_B}}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/output/fiq_rerank/blip2_qtr}"
EPOCHS="${EPOCHS:-40}"
VAL_EVERY="${VAL_EVERY:-2}"
LR="${LR:-2e-5}"
K_VAL="${K_VAL:-100}"

case "${OUT_ROOT}" in *cirr*|*CIRR*) echo "ABORT: FIQ must not write under cirr: ${OUT_ROOT}" >&2; exit 2 ;; esac

cd "${ROOT}"
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
  --k_val "${K_VAL}"
  --early_stop_patience 6
  --fiq_categories dress,shirt,toptee
  --topk_dir "${ROOT}/output/fiq_vision_desc"
  "$@"
)

echo "============================================================"
echo "FIQ QTR Stage-A | exp=${EXP_NAME}"
echo "Backbone=${PRETRAINED_ID} | local_bs=${BATCH_SIZE} ×${NGPU} => global_CE=${GLOBAL_B}"
echo "Val: HBI Top-${K_VAL} pool mean R@10 | Out: ${OUT_ROOT}/${EXP_NAME}"
echo "============================================================"

mkdir -p "${OUT_ROOT}"
if [[ "$NGPU" -gt 1 ]]; then
  exec "${VENV_PY}" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    HBI/fiq/vision_desc/train_fiq_blip2_qtr.py "${ARGS[@]}"
else
  exec "${VENV_PY}" HBI/fiq/vision_desc/train_fiq_blip2_qtr.py "${ARGS[@]}"
fi
