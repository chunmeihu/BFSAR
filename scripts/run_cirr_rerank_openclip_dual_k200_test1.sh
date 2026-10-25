#!/usr/bin/env bash
# CIRR test1: OpenCLIP dual coarse (val indep-best) Top-200 + Composer + CR
# + frozen val fusion weights 0.20/0.35/0.45 → official submission JSONs.
#
# Do NOT retune αβw / fusion weights on test1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${VENV_PY:-${ROOT}/.venv/bin/python}"
export VIRTUAL_ENV="${ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CKPT="${INIT_MODEL:-${ROOT}/output/cirr_retrieval_openclip/pytorch_model.bin.best.9}"
DUAL_JSON="${DUAL_JSON:-${ROOT}/data/cirr/vision_desc/test1_dual_targets.json}"
CLIP_CKPT="${CLIP_CKPT:-${ROOT}/weights/open_clip/hbi_format/ViT-L-14-laion2B-s32B-b82K.pt}"
STAGEA_OUT="${STAGEA_OUT:-${ROOT}/output/cirr_vision_desc/sdr_wti_native_openclip_test1}"
OUT="${OUT_DIR:-${ROOT}/output/cirr_rerank/fusion_openclip_dual_k200_test1}"
COMPOSER="${COMPOSER_CKPT:-${ROOT}/output/cirr_rerank/blip2_qtr/qtr_v1_stageA_g32/qtr_stagea_best_pool_r1.pt}"
STAGE1="${STAGE1_CKPT:-${ROOT}/weights/cr/stage1/CIRR/saved_models/blip_mean.pt}"
STAGE2="${STAGE2_CKPT:-${ROOT}/weights/cr/stage2/CIRR/saved_models/blip_mean.pt}"
K="${K_RERANK:-200}"
# val zscore best fuse (OpenCLIP dual + Composer + CR)
ALPHA_HBI="${ALPHA_HBI:-0.20}"
BETA_COMPOSER="${BETA_COMPOSER:-0.35}"
GAMMA_CR="${GAMMA_CR:-0.45}"
SUB_NAME="${SUBMISSION_NAME:-openclip_dual_composer_cr_k200}"
LOGDIR="${OUT}/logs"
mkdir -p "${STAGEA_OUT}" "${OUT}" "${LOGDIR}" "${ROOT}/output/cirr_vision_desc/submission_CIRR"

abort() { echo "ABORT: $*" >&2; exit 1; }
case "${OUT}${STAGEA_OUT}" in
  *openclip*) ;;
  *) abort "paths must contain openclip" ;;
esac
for f in "${CKPT}" "${DUAL_JSON}" "${CLIP_CKPT}" "${COMPOSER}" "${STAGE1}" "${STAGE2}"; do
  [[ -f "${f}" ]] || abort "missing ${f}"
done

COMMON=(
  -u HBI/cirr/vision_desc/eval_hbi_composer_cr_fusion_test1.py
  --topk_pt "${OUT}/cirr_top_${K}_test1.pt"
  --hbi_npz "${OUT}/sim_hbi.npz"
  --composer_ckpt "${COMPOSER}"
  --stage1_path "${STAGE1}"
  --stage2_path "${STAGE2}"
  --out_dir "${OUT}"
  --k_rerank "${K}"
  --alpha_hbi "${ALPHA_HBI}"
  --beta_composer "${BETA_COMPOSER}"
  --gamma_cr "${GAMMA_CR}"
  --submission_name "${SUB_NAME}"
  --num_workers 0
)

{
  echo "============================================================"
  echo "CIRR test1 OpenCLIP dual + Composer + CR | K=${K}"
  echo "started $(date -Iseconds)"
  echo "ckpt=${CKPT}"
  echo "dual=${DUAL_JSON}"
  echo "fusion weights HBI/Composer/CR=${ALPHA_HBI}/${BETA_COMPOSER}/${GAMMA_CR}"
  echo "submission_name=${SUB_NAME}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"
} | tee "${LOGDIR}/pipeline.log"

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  abort "nvidia-smi failed; run from a GPU-visible shell"
fi
"${VENV_PY}" - <<'PY' || abort "CUDA not available in venv"
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY

echo "[1/5] Stage-A OpenCLIP branch sims (test1) $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" -u -m HBI.cirr.vision_desc.compute_cirr_openclip_test1_dual_branches \
  --init_model "${CKPT}" \
  --dual_json "${DUAL_JSON}" \
  --clip_checkpoint "${CLIP_CKPT}" \
  --output_npz "${STAGEA_OUT}/branch_sims.npz" \
  --device "cuda:0" \
  --recompute \
  --batch_size_val 8 \
  --query_batch 32 \
  --gallery_batch 64 \
  --workers 4 \
  2>&1 | tee "${LOGDIR}/stageA.log" | tee -a "${LOGDIR}/pipeline.log"

echo "[2/5] dump dual sim + Top-${K} (CPU) $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" -u -m HBI.cirr.vision_desc.dump_cirr_openclip_dual_test1 \
  --branch_npz "${STAGEA_OUT}/branch_sims.npz" \
  --out_dir "${OUT}" \
  --k_save "${K}" \
  2>&1 | tee "${LOGDIR}/dump.log" | tee -a "${LOGDIR}/pipeline.log"

echo "[3/5] Composer ITC test1 $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" "${COMMON[@]}" --only composer \
  2>&1 | tee "${LOGDIR}/composer.log" | tee -a "${LOGDIR}/pipeline.log"

echo "[4/5] CR NLVR test1 $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" "${COMMON[@]}" --only cr \
  2>&1 | tee "${LOGDIR}/cr.log" | tee -a "${LOGDIR}/pipeline.log"

echo "[5/5] fuse + write submission $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" "${COMMON[@]}" --only fuse \
  2>&1 | tee "${LOGDIR}/fuse.log" | tee -a "${LOGDIR}/pipeline.log"

SUB_DIR="${ROOT}/output/cirr_vision_desc/submission_CIRR"
echo "=== submission files ===" | tee -a "${LOGDIR}/pipeline.log"
ls -lh "${SUB_DIR}/recall_submission_${SUB_NAME}.json" \
       "${SUB_DIR}/recall_subset_submission_${SUB_NAME}.json" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" - <<PY | tee -a "${LOGDIR}/pipeline.log"
import json
from pathlib import Path
meta = json.loads(Path("${OUT}/test1_fusion_meta.json").read_text())
print(json.dumps(meta, indent=2, ensure_ascii=False))
rec = json.loads(Path("${SUB_DIR}/recall_submission_${SUB_NAME}.json").read_text())
sub = json.loads(Path("${SUB_DIR}/recall_subset_submission_${SUB_NAME}.json").read_text())
n = len([k for k in rec if k not in ("version", "metric")])
pid = next(k for k in rec if k not in ("version", "metric"))
print(f"recall n={n} top50_len={len(rec[pid])} subset_len={len(sub[pid])} version={rec.get('version')}")
PY

echo "DONE $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
echo "Upload to CIRR server:"
echo "  ${SUB_DIR}/recall_submission_${SUB_NAME}.json"
echo "  ${SUB_DIR}/recall_subset_submission_${SUB_NAME}.json"
