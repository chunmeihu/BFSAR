#!/usr/bin/env bash
# Isolated FIQ ablation train (A2/A3). Writes ONLY under output/ablations/fiq_coarse/.
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
OUT_DIR="${OUT_DIR:-${ROOT}/output/ablations/fiq_coarse/${ABLATION_ID}_train}"
NGPU="${NGPU:-2}"
MASTER_PORT="${MASTER_PORT:-29652}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
WARMUP="${WARMUP:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WTI_LEVELS="${WTI_LEVELS:-3}"
SKL="${SKL:-1.0}"
ESTIMATOR="${ESTIMATOR:-${ROOT}/output/fiq_estimator_openclip/banzhaf_estimator.epoch10.pt}"

cd "${ROOT}"
case "${OUT_DIR}" in
  */output/ablations/*) ;;
  *) echo "ABORT: OUT_DIR must be under output/ablations/: ${OUT_DIR}" >&2; exit 2 ;;
esac
case "${OUT_DIR}" in
  *fiq_retrieval_openclip*|*fiq_estimator_openclip*|*refsim_fusion*|*sdr_dual*|*fusion_k*)
    echo "ABORT: refusing mainline path: ${OUT_DIR}" >&2; exit 2 ;;
esac

EXTRA_ARGS=()
if [[ "${ESTIMATOR}" == "None" || "${ESTIMATOR}" == "none" || -z "${ESTIMATOR}" ]]; then
  ESTIMATOR="None"
else
  if [[ ! -f "${ESTIMATOR}" ]]; then
    echo "ABORT: missing estimator: ${ESTIMATOR}" >&2
    exit 2
  fi
fi

mkdir -p "${OUT_DIR}"
{
  echo "exp=fiq_coarse_ablation_${ABLATION_ID}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "ESTIMATOR=${ESTIMATOR}"
  echo "SKL=${SKL}"
  echo "WTI_LEVELS=${WTI_LEVELS}"
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
  --fiq_categories "dress,shirt,toptee"
  --max_frames 12
  --max_words 32
  --v_rate0 0.25
  --v_rate1 0.5
  --image_encode_microbatch 12
)

echo "============================================================"
echo "FIQ ablation ${ABLATION_ID} | out=${OUT_DIR}"
echo "estimator=${ESTIMATOR} skl=${SKL} wti_levels=${WTI_LEVELS}"
echo "============================================================"

if [[ "${NGPU}" -gt 1 ]]; then
  exec "${VENV_PY}" -m torch.distributed.run --standalone --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT}" \
    main_fiq_hbi_retrieval_openclip.py "${ARGS[@]}" "$@"
else
  exec "${VENV_PY}" main_fiq_hbi_retrieval_openclip.py "${ARGS[@]}" "$@"
fi
