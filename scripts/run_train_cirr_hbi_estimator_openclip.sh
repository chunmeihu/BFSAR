#!/usr/bin/env bash
# CIRR HBI Stage-1: Banzhaf estimator with OpenCLIP ViT-L-14 (laion2b_s32b_b82k).
# Same hyperparams as original CIRR estimator (12-crop, max_words=32, MC=50).
# Writes to output/cirr_estimator_openclip — does not overwrite OpenAI CIRR runs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
export VIRTUAL_ENV="${ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export ASSET_ROOT="${ASSET_ROOT:-$HOME}"
export PYTHONPATH="${ROOT}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER 2>/dev/null || true

NGPU="${NGPU:-2}"
if [[ "${NGPU}" -gt 1 ]]; then
  IFS=',' read -r -a _vis <<< "${CUDA_VISIBLE_DEVICES}"
  if [[ "${#_vis[@]}" -lt "${NGPU}" ]]; then
    echo "ABORT: NGPU=${NGPU} but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (need ${NGPU} devices)" >&2
    exit 2
  fi
fi
MASTER_PORT="${MASTER_PORT:-29561}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-2}"
OUT_DIR="${OUT_DIR:-${ROOT}/output/cirr_estimator_openclip}"
CLIP_CKPT="${CLIP_CKPT:-${ROOT}/weights/open_clip/hbi_format/ViT-L-14-laion2B-s32B-b82K.pt}"

cd "${ROOT}"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "ABORT: missing venv python: ${VENV_PY}" >&2
  exit 2
fi
case "${OUT_DIR}" in
  *fiq*) echo "ABORT: CIRR run must not write under a fiq path: ${OUT_DIR}" >&2; exit 2 ;;
esac
if [[ "${OUT_DIR}" != *openclip* ]]; then
  echo "ABORT: CIRR OpenCLIP output_dir must contain 'openclip': ${OUT_DIR}" >&2
  exit 2
fi
if [[ "${OUT_DIR}" == "${ROOT}/output/cirr_retrieval" ]]; then
  echo "ABORT: refusing original OpenAI CIRR retrieval dir" >&2
  exit 2
fi
if [[ ! -f "${CLIP_CKPT}" ]]; then
  echo "ABORT: missing OpenCLIP HBI-format weights: ${CLIP_CKPT}" >&2
  exit 2
fi

"${VENV_PY}" - <<'PY'
import os, subprocess, sys
print("python", sys.executable)
print("VIRTUAL_ENV", os.environ.get("VIRTUAL_ENV"))
print("PYTHONPATH", os.environ.get("PYTHONPATH"))
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
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

echo "============================================================"
echo "CIRR Banzhaf estimator OpenCLIP-L | out=${OUT_DIR}"
echo "local_bs=${BATCH_SIZE} ×${NGPU} accum=${GRAD_ACCUM} epochs=${EPOCHS} lr=${LR}"
echo "============================================================"

ARGS=(
  --output_dir "${OUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum_steps "${GRAD_ACCUM}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --workers "${NUM_WORKERS}"
  --max_frames 12
  --max_words 32
  --banzhaf_mc_samples 50
  --clip_checkpoint "${CLIP_CKPT}"
)

if [[ -n "${CIRR_IMAGE_ROOT:-}" ]]; then
  ARGS+=(--cirr_image_root "${CIRR_IMAGE_ROOT}")
fi

mkdir -p "${OUT_DIR}"
{
  echo "exp=cirr_hbi_estimator_openclip"
  echo "clip=open_clip ViT-L-14 / laion2b_s32b_b82k"
  echo "python=${VENV_PY}"
  echo "VIRTUAL_ENV=${VIRTUAL_ENV}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "NGPU=${NGPU}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "CLIP_CKPT=${CLIP_CKPT}"
  echo "max_frames=12"
  echo "max_words=32"
  echo "banzhaf_mc_samples=50"
  echo "started=$(date -Iseconds)"
} > "${OUT_DIR}/run_env.txt"

if [[ "$NGPU" -gt 1 ]]; then
  exec "${VENV_PY}" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    main_cirr_hbi_estimator_openclip.py "${ARGS[@]}" "$@"
else
  exec "${VENV_PY}" main_cirr_hbi_estimator_openclip.py "${ARGS[@]}" "$@"
fi
