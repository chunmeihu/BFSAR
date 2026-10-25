#!/usr/bin/env bash
# FIQ A3: single-layer Banzhaf coarse retriever + dual-path coarse + Composer/CR rerank.
# Large gallery = Original-Split; small gallery = VAL-Split (MCoT).
# Dump is CPU (runs now). Composer/CR wait for a free GPU so they do not
# sit on leftover VRAM of CIRR A3s 2×16 training.
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
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER 2>/dev/null || true

WTI="${ROOT}/output/ablations/fiq_coarse/A3_entity_only_dual"
CKPT="${ROOT}/output/ablations/fiq_coarse/A3_entity_only/pytorch_model.bin.best.4"
COMPOSER="${ROOT}/output/fiq_rerank/blip2_qtr/qtr_v1_stageA_g32/qtr_stagea_best_pool_r10.pt"
STAGE1="${ROOT}/weights/cr/stage1/fashionIQ/saved_models/blip.pt"
STAGE2="${ROOT}/weights/cr/stage2/fashionIQ/saved_models/blip.pt"
OUT_LARGE="${ROOT}/output/ablations/fiq_rerank/A3_dual_k300_originalsplit"
OUT_SMALL="${ROOT}/output/ablations/fiq_rerank/A3_dual_k300_valsplit"
SWEEP="${ROOT}/output/ablations/fiq_rerank/A3_dual_rerank_sliced"
LOGDIR="${ROOT}/output/ablations/logs"
K=300
mkdir -p "${LOGDIR}" "${OUT_LARGE}" "${OUT_SMALL}" "${SWEEP}"

case "${OUT_LARGE}${OUT_SMALL}${SWEEP}" in
  */output/fiq_rerank/fusion_k*|*/output/fiq_rerank/ksweep_*)
    echo "ABORT: refusing mainline fiq_rerank path" >&2; exit 2 ;;
esac
for f in "${WTI}/scores_dress.npz" "${CKPT}" "${COMPOSER}" "${STAGE1}" "${STAGE2}"; do
  [[ -f "${f}" ]] || { echo "ABORT: missing ${f}" >&2; exit 2; }
done

wait_free_gpu() {
  local need="${1:-18000}"
  while true; do
    local i free
    for i in 0 1; do
      free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${i}" | awk '{print int($1)}')"
      if [[ "${free}" -ge "${need}" ]]; then
        echo "[A3 rerank] GPU${i} free=${free}MiB (>=${need}); take this card"
        export CUDA_VISIBLE_DEVICES="${i}"
        return 0
      fi
    done
    echo "[A3 rerank] wait GPU free>=${need}MiB $(date -Iseconds)"
    sleep 60
  done
}

score_gpu() {
  local out="$1"
  local common=(
    -u -m HBI.fiq.vision_desc.eval_fiq_hbi_composer_cr_fusion
    --init_model "${CKPT}"
    --composer_ckpt "${COMPOSER}"
    --stage1_ckpt "${STAGE1}"
    --stage2_ckpt "${STAGE2}"
    --out_dir "${out}"
    --k_save "${K}"
    --k_rerank "${K}"
  )
  if [[ ! -f "${out}/composer_toptee.npz" ]]; then
    echo "[composer] ${out} $(date -Iseconds)"
    "${VENV_PY}" "${common[@]}" --only composer
  else
    echo "[skip composer] ${out}"
  fi
  if [[ ! -f "${out}/cr_toptee.npz" ]]; then
    echo "[cr] ${out} $(date -Iseconds)"
    "${VENV_PY}" "${common[@]}" --only cr
  else
    echo "[skip cr] ${out}"
  fi
}

echo "============================================================"
echo "FIQ A3 single-layer Banzhaf dual + Composer/CR rerank | large+small gallery"
echo "started $(date -Iseconds)"
echo "============================================================"

if [[ ! -f "${OUT_LARGE}/hbi_top${K}_val_dress.pt" ]]; then
  echo "[dump original-split Top-${K} CPU] $(date -Iseconds)"
  "${VENV_PY}" -u -m HBI.fiq.vision_desc.dump_fiq_openclip_dual_topk \
    --wti_dir "${WTI}" \
    --out_dir "${OUT_LARGE}" \
    --k_save "${K}" \
    --alpha 0.4 --beta 0.2 --w_ref 0.5
else
  echo "[skip dump original-split]"
fi

if [[ ! -f "${OUT_SMALL}/hbi_top${K}_val_dress.pt" ]]; then
  echo "[dump val-split Top-${K} CPU] $(date -Iseconds)"
  "${VENV_PY}" -u -m ablations.coarse_retrieval.dump_fiq_a3_valsplit_topk \
    --wti_dir "${WTI}" \
    --out_dir "${OUT_SMALL}" \
    --k_save "${K}" \
    --alpha 0.4 --beta 0.2 --w_ref 0.5
else
  echo "[skip dump val-split]"
fi

wait_free_gpu 18000
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $(date -Iseconds)"
score_gpu "${OUT_LARGE}"
score_gpu "${OUT_SMALL}"

echo "[ksweep val-split K=300 CPU] $(date -Iseconds)"
"${VENV_PY}" -u -m HBI.fiq.vision_desc.eval_fiq_valsplit_ksweep \
  --score_dir "${OUT_SMALL}" \
  --out_json "${OUT_SMALL}/valsplit_k300.json" \
  --ks 300

echo "[ksweep original-split K=300 CPU] $(date -Iseconds)"
"${VENV_PY}" -u -m HBI.fiq.vision_desc.eval_fiq_originalsplit_ksweep \
  --score_dir "${OUT_LARGE}" \
  --out_dir "${SWEEP}/originalsplit" \
  --per_k_out_root "${SWEEP}" \
  --valsplit_json "${OUT_SMALL}/valsplit_k300.json" \
  --ks 300 \
  --no_grid

"${VENV_PY}" - <<PY
import json
from pathlib import Path
root = Path("${SWEEP}")
large = json.loads((root / "originalsplit" / "originalsplit_k200_300_500.json").read_text())
small = json.loads(Path("${OUT_SMALL}/valsplit_k300.json").read_text())
cats = ("dress", "shirt", "toptee")

def large_row(k):
    blk = large["by_k"][str(k)]
    per = {}
    # originalsplit summary compact: look at per-k fusion_metrics
    mpath = root / f"fusion_k{k}_openclip_dual" / "fusion_metrics.json"
    m = json.loads(mpath.read_text())
    out = {"mean_R@10": m["mean"]["frozen_0.25_0.30_0.45"]["recall_at10"],
           "mean_R@50": m["mean"]["frozen_0.25_0.30_0.45"]["recall_at50"]}
    for c in cats:
        f = m["per_category"][c]["frozen_0.25_0.30_0.45_full"]
        out[c] = {"R@10": f["recall_at10"], "R@50": f["recall_at50"],
                  "n_gallery": m["per_category"][c]["n_gallery"]}
    return out

def small_row(k):
    blk = small["by_k"][str(k)]
    out = {"mean_R@10": blk["mean"]["mcot_no_drop"]["recall_at10"],
           "mean_R@50": blk["mean"]["mcot_no_drop"]["recall_at50"]}
    for c in cats:
        f = blk["per_category"][c]["mcot_no_drop"]
        out[c] = {"R@10": f["recall_at10"], "R@50": f["recall_at50"],
                  "n_gallery": blk["per_category"][c]["n_gallery"]}
    return out

report = {
    "coarse": "A3 single-layer Banzhaf WTI + dual Path_ref/Path_vis α=0.4 β=0.2 w_ref=0.5",
    "rerank": "Composer BLIP2-QTR + CR BLIP, frozen 0.25/0.30/0.45, K=300 (same as 3-level SOTA)",
    "large_original_split_K300": large_row(300),
    "small_val_split_K300": small_row(300),
}
outp = Path("${ROOT}/output/ablations/fiq_rerank/A3_dual_rerank_final.json")
outp.write_text(json.dumps(report, indent=2) + "\n")
L = report["large_original_split_K300"]
S = report["small_val_split_K300"]
print("=== FIQ A3 dual + rerank FINAL (K=300) ===")
print("Large gallery (Original-Split, drop-ref)")
for c in cats:
    print(f"  {c:8s}  R@10={L[c]['R@10']:.2f}  R@50={L[c]['R@50']:.2f}  gallery={L[c]['n_gallery']}")
print(f"  mean      R@10={L['mean_R@10']:.2f}  R@50={L['mean_R@50']:.2f}")
print("Small gallery (VAL-Split, MCoT no-drop)")
for c in cats:
    print(f"  {c:8s}  R@10={S[c]['R@10']:.2f}  R@50={S[c]['R@50']:.2f}  gallery={S[c]['n_gallery']}")
print(f"  mean      R@10={S['mean_R@10']:.2f}  R@50={S['mean_R@50']:.2f}")
print("wrote", outp)
PY

echo "OK: FIQ A3 dual rerank done." | tee "${LOGDIR}/fiq_a3_rerank_done.txt"
echo "finished=$(date -Iseconds)"
