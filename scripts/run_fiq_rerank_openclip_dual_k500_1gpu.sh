#!/usr/bin/env bash
# FIQ val rerank on OpenCLIP dual-path coarse Top-500, single GPU.
# Composer = BLIP2-QTR, CR = Candidate-Reranking BLIP (no CLIP).
# Sequential: dump(CPU) -> composer -> CR -> fuse -> boost.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${ROOT}/.venv/bin/python"
export VIRTUAL_ENV="${ROOT}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export ASSET_ROOT="${ASSET_ROOT:-$HOME}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT="${OUT_DIR:-${ROOT}/output/fiq_rerank/fusion_k500_openclip_dual}"
WTI="${WTI_DIR:-${ROOT}/output/fiq_vision_desc/refsim_fusion_openclip}"
# init_model only for argparse / fuse meta; Composer/CR do not load CLIP/HBI.
CKPT="${INIT_MODEL:-${ROOT}/output/fiq_retrieval_openclip/pytorch_model.bin.best.3}"
COMPOSER="${COMPOSER_CKPT:-${ROOT}/output/fiq_rerank/blip2_qtr/qtr_v1_stageA_g32/qtr_stagea_best_pool_r10.pt}"
STAGE1="${STAGE1_CKPT:-${ROOT}/weights/cr/stage1/fashionIQ/saved_models/blip.pt}"
STAGE2="${STAGE2_CKPT:-${ROOT}/weights/cr/stage2/fashionIQ/saved_models/blip.pt}"
K="${K_RERANK:-500}"
LOGDIR="${ROOT}/output/fiq_rerank/logs_openclip_dual"
mkdir -p "${OUT}" "${LOGDIR}"

case "${OUT}${WTI}${CKPT}${COMPOSER}${STAGE1}${STAGE2}" in
  *cirr*|*CIRR*) echo "ABORT: cirr path" >&2; exit 2 ;;
esac
for f in "${WTI}/scores_dress.npz" "${COMPOSER}" "${STAGE1}" "${STAGE2}" "${CKPT}"; do
  [[ -f "${f}" ]] || { echo "ABORT: missing ${f}" >&2; exit 2; }
done

COMMON=(
  -u -m HBI.fiq.vision_desc.eval_fiq_hbi_composer_cr_fusion
  --init_model "${CKPT}"
  --composer_ckpt "${COMPOSER}"
  --stage1_ckpt "${STAGE1}"
  --stage2_ckpt "${STAGE2}"
  --out_dir "${OUT}"
  --k_save "${K}"
  --k_rerank "${K}"
)

{
  echo "============================================================"
  echo "FIQ OpenCLIP dual coarse Top-${K} + Composer + CR | 1-GPU"
  echo "started $(date -Iseconds)"
  echo "out=${OUT}"
  echo "wti=${WTI}"
  echo "composer=${COMPOSER} (BLIP2-QTR, not CLIP)"
  echo "cr=${STAGE1} + ${STAGE2} (BLIP CR, not CLIP)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"
} | tee "${LOGDIR}/pipeline.log"

echo "[1/5] dump dual-path Top-${K} (CPU)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" -u -m HBI.fiq.vision_desc.dump_fiq_openclip_dual_topk \
  --wti_dir "${WTI}" \
  --out_dir "${OUT}" \
  --k_save "${K}" \
  --alpha 0.4 --beta 0.2 --w_ref 0.5 \
  2>&1 | tee -a "${LOGDIR}/pipeline.log"

echo "[2/5] Composer on GPU${CUDA_VISIBLE_DEVICES} $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" "${COMMON[@]}" --only composer \
  2>&1 | tee -a "${LOGDIR}/composer.log" | tee -a "${LOGDIR}/pipeline.log"

echo "[3/5] CR on GPU${CUDA_VISIBLE_DEVICES} $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" "${COMMON[@]}" --only cr \
  2>&1 | tee -a "${LOGDIR}/cr.log" | tee -a "${LOGDIR}/pipeline.log"

echo "[4/5] fuse $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" "${COMMON[@]}" --only fuse \
  2>&1 | tee -a "${LOGDIR}/fuse.log" | tee -a "${LOGDIR}/pipeline.log"

echo "[5/5] boost $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" -u -m HBI.fiq.vision_desc.eval_fiq_fusion_boost --out_dir "${OUT}" \
  2>&1 | tee -a "${LOGDIR}/boost.log" | tee -a "${LOGDIR}/pipeline.log"

"${VENV_PY}" - <<PY | tee -a "${LOGDIR}/pipeline.log"
import json
from pathlib import Path
d = Path("${OUT}")
fus = json.loads((d / "fusion_metrics.json").read_text())["mean"]
print("=== fusion mean ===")
print("frozen", {k: round(fus["frozen_0.25_0.30_0.45"][k], 2) for k in ("recall_at10","recall_at50","recall_at100","recall_at200")})
print("grid_best", {k: round(fus["grid_best_r10"][k], 2) for k in ("recall_at10","recall_at50","recall_at100","recall_at200")},
      "w=", fus["grid_best_r10"].get("alpha_coarse"), fus["grid_best_r10"].get("beta_composer"), fus["grid_best_r10"].get("gamma_cr"))
boost = d / "boost_metrics.json"
if boost.exists():
    b = json.loads(boost.read_text())["mean"]
    print("=== boost mean R@10 ===")
    for k,v in b.items():
        print(f"  {k}: {v['recall_at10']:.2f}")
print("prev k500 compose41 grid_best R@10 ~58.35 | MCoT paper ~63.24")
PY

echo "DONE $(date -Iseconds) -> ${OUT}/fusion_metrics.json" | tee -a "${LOGDIR}/pipeline.log"
