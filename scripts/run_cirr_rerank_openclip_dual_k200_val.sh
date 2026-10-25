#!/usr/bin/env bash
# CIRR val: OpenCLIP dual-path coarse Top-200 + Composer ITC + CR NLVR + 3-way fuse.
# Same rerank recipe as fusion_hbi_composer_cr / k200, only coarse is dual indep-best.
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

OUT="${OUT_DIR:-${ROOT}/output/cirr_rerank/fusion_openclip_dual_k200_val}"
BRANCH_NPZ="${BRANCH_NPZ:-${ROOT}/output/cirr_vision_desc/sdr_wti_native_openclip_val/branch_sims.npz}"
DUAL_METRICS="${DUAL_METRICS:-${ROOT}/output/cirr_vision_desc/sdr_dual_sr_ref_vision_openclip_val/metrics.json}"
COMPOSER="${COMPOSER_CKPT:-${ROOT}/output/cirr_rerank/blip2_qtr/qtr_v1_stageA_g32/qtr_stagea_best_pool_r1.pt}"
STAGE1="${STAGE1_CKPT:-${ROOT}/weights/cr/stage1/CIRR/saved_models/blip_mean.pt}"
STAGE2="${STAGE2_CKPT:-${ROOT}/weights/cr/stage2/CIRR/saved_models/blip_mean.pt}"
K="${K_RERANK:-200}"
LOGDIR="${LOGDIR:-${OUT}/logs}"
mkdir -p "${OUT}" "${LOGDIR}"

case "${OUT}" in
  *openclip*dual*) ;;
  *) echo "ABORT: OUT must contain openclip+dual isolation: ${OUT}" >&2; exit 2 ;;
esac

for f in "${BRANCH_NPZ}" "${DUAL_METRICS}" "${COMPOSER}" "${STAGE1}" "${STAGE2}"; do
  [[ -f "${f}" ]] || { echo "ABORT: missing ${f}" >&2; exit 2; }
done

COMMON=(
  -u HBI/cirr/vision_desc/eval_hbi_composer_cr_fusion.py
  --topk_pt "${OUT}/cirr_top_${K}_val.pt"
  --hbi_npz "${OUT}/sim_hbi.npz"
  --composer_ckpt "${COMPOSER}"
  --stage1_path "${STAGE1}"
  --stage2_path "${STAGE2}"
  --out_dir "${OUT}"
  --k_rerank "${K}"
  --num_workers 0
)

{
  echo "============================================================"
  echo "CIRR OpenCLIP dual coarse Top-${K} + Composer + CR | GPU=${CUDA_VISIBLE_DEVICES}"
  echo "started $(date -Iseconds)"
  echo "out=${OUT}"
  echo "branch=${BRANCH_NPZ}"
  echo "composer=${COMPOSER}"
  echo "cr=${STAGE1} + ${STAGE2}"
  echo "============================================================"
} | tee "${LOGDIR}/pipeline.log"

echo "[1/4] dump dual sim + Top-${K} (CPU) $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" -u -m HBI.cirr.vision_desc.dump_cirr_openclip_dual_sim \
  --branch_npz "${BRANCH_NPZ}" \
  --metrics_json "${DUAL_METRICS}" \
  --out_dir "${OUT}" \
  --k_save "${K}" \
  2>&1 | tee -a "${LOGDIR}/dump.log" | tee -a "${LOGDIR}/pipeline.log"

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  echo "ABORT: nvidia-smi failed. Dump done; re-run this script from a GPU shell for Composer/CR." >&2
  exit 1
fi

echo "[2/4] Composer ITC $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" "${COMMON[@]}" --only composer \
  2>&1 | tee "${LOGDIR}/composer.log" | tee -a "${LOGDIR}/pipeline.log"

echo "[3/4] CR NLVR $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" "${COMMON[@]}" --only cr \
  2>&1 | tee "${LOGDIR}/cr.log" | tee -a "${LOGDIR}/pipeline.log"

echo "[4/4] fuse grid $(date -Iseconds)" | tee -a "${LOGDIR}/pipeline.log"
"${VENV_PY}" "${COMMON[@]}" --only fuse \
  2>&1 | tee "${LOGDIR}/fuse.log" | tee -a "${LOGDIR}/pipeline.log"

"${VENV_PY}" - <<PY | tee -a "${LOGDIR}/pipeline.log"
import json
from pathlib import Path
p = Path("${OUT}") / "fusion_metrics.json"
d = json.loads(p.read_text())
z = d["zscore"]
print("=== CIRR OpenCLIP dual + Composer + CR (val, K=${K}) ===")
print("n_queries", d["n_queries"], "hbi_missing_rows", d["hbi_missing_rows"])
print("singles:")
for name, m in z["singles"].items():
    print(f"  {name:10s} R@1={m['pool_recall_at1']:.2f} R@5={m['pool_recall_at5']:.2f} "
          f"R@10={m['pool_recall_at10']:.2f} R@50={m['pool_recall_at50']:.2f}")
b = z["best"]
print("best fuse (zscore):")
print(f"  α_hbi={b['alpha_hbi']} β_comp={b['beta_composer']} γ_cr={b['gamma_cr']}")
print(f"  R@1={b['pool_recall_at1']:.2f} R@5={b['pool_recall_at5']:.2f} "
      f"R@10={b['pool_recall_at10']:.2f} R@50={b['pool_recall_at50']:.2f}")
if "pool_group_recall_at1" in b:
    print(f"  G-R@1={b['pool_group_recall_at1']:.2f} G-R@2={b.get('pool_group_recall_at2',0):.2f} "
          f"G-R@3={b.get('pool_group_recall_at3',0):.2f}")
print(f"vs CR paper 53.36: ΔR@1={d.get('delta_vs_cr', b['pool_recall_at1']-53.36):+.2f}")
print("wrote", p)
PY

echo "DONE $(date -Iseconds) -> ${OUT}/fusion_metrics.json" | tee -a "${LOGDIR}/pipeline.log"
