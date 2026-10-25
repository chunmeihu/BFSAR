#!/usr/bin/env bash
# Isolated CIRR ablation train (A2/A3). Writes ONLY under output/ablations/cirr_coarse/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
export VIRTUAL_ENV="${ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export ASSET_ROOT="${ASSET_ROOT:-$HOME}"
export PYTHONPATH="${ROOT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER 2>/dev/null || true

ABLATION_ID="${ABLATION_ID:?set ABLATION_ID=A2|A3}"
OUT_DIR="${OUT_DIR:-${ROOT}/output/ablations/cirr_coarse/${ABLATION_ID}_train}"
NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29662}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-16}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WTI_LEVELS="${WTI_LEVELS:-3}"
SKL="${SKL:-1.0}"
ESTIMATOR="${ESTIMATOR:-${ROOT}/output/cirr_estimator_openclip/banzhaf_estimator.epoch10.pt}"
CLIP_CKPT="${CLIP_CKPT:-${ROOT}/weights/open_clip/hbi_format/ViT-L-14-laion2B-s32B-b82K.pt}"

cd "${ROOT}"
case "${OUT_DIR}" in
  */output/ablations/*) ;;
  *) echo "ABORT: OUT_DIR must be under output/ablations/: ${OUT_DIR}" >&2; exit 2 ;;
esac
case "${OUT_DIR}" in
  *cirr_retrieval_openclip*|*cirr_estimator_openclip*|*sdr_wti_native*|*sdr_dual*|*fusion_openclip_dual*)
    echo "ABORT: refusing mainline path: ${OUT_DIR}" >&2; exit 2 ;;
esac

if [[ "${ESTIMATOR}" == "None" || "${ESTIMATOR}" == "none" || -z "${ESTIMATOR}" ]]; then
  ESTIMATOR="None"
else
  if [[ ! -f "${ESTIMATOR}" ]]; then
    echo "ABORT: missing estimator: ${ESTIMATOR}" >&2
    exit 2
  fi
fi
if [[ ! -f "${CLIP_CKPT}" ]]; then
  echo "ABORT: missing clip ckpt: ${CLIP_CKPT}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
{
  echo "exp=cirr_coarse_ablation_${ABLATION_ID}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "ESTIMATOR=${ESTIMATOR}"
  echo "SKL=${SKL}"
  echo "WTI_LEVELS=${WTI_LEVELS}"
  echo "NGPU=${NGPU}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "BATCH_SIZE_VAL=${BATCH_SIZE_VAL}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "started=$(date -Iseconds)"
} > "${OUT_DIR}/run_env.txt"

ARGS=(
  --do_train 1
  --output_dir "${OUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --batch_size_val "${BATCH_SIZE_VAL}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --warmup_proportion "${WARMUP}"
  --workers "${NUM_WORKERS}"
  --estimator "${ESTIMATOR}"
  --skl "${SKL}"
  --wti_levels "${WTI_LEVELS}"
  --max_frames 12
  --max_words 32
  --v_rate0 0.25
  --v_rate1 0.5
  --image_encode_microbatch 12
  --clip_checkpoint "${CLIP_CKPT}"
)

echo "============================================================"
echo "CIRR ablation ${ABLATION_ID} | out=${OUT_DIR}"
echo "estimator=${ESTIMATOR} skl=${SKL} wti_levels=${WTI_LEVELS}"
echo "============================================================"

if [[ "${NGPU}" -gt 1 ]]; then
  exec "${VENV_PY}" -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT}" \
    main_cirr_hbi_retrieval_openclip.py "${ARGS[@]}" "$@"
else
  exec "${VENV_PY}" main_cirr_hbi_retrieval_openclip.py "${ARGS[@]}" "$@"
fi
