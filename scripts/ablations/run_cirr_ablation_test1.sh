#!/usr/bin/env bash
# CIRR ablation on test1: encode (wti=1) → MIC/dual CodaLab zips → Top-200
# dump → Composer ITC + CR NLVR → frozen 0.20/0.35/0.45 fuse zip.
# Already-finished val runs are left as-is. Writes only under output/ablations/.
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

TAG="${1:?usage: run_cirr_ablation_test1.sh A3|A2e|A3s|A3t|A3y [--encode-only|--dump-only|--coarse-only]}"
MODE="${2:-}"

K=200
COMPOSER="${ROOT}/output/cirr_rerank/blip2_qtr/qtr_v1_stageA_g32/qtr_stagea_best_pool_r1.pt"
STAGE1="${ROOT}/weights/cr/stage1/CIRR/saved_models/blip_mean.pt"
STAGE2="${ROOT}/weights/cr/stage2/CIRR/saved_models/blip_mean.pt"
COARSE_ROOT="${ROOT}/output/ablations/cirr_coarse_test1"
LOGDIR="${ROOT}/output/ablations/logs"
mkdir -p "${LOGDIR}" "${COARSE_ROOT}"

case "${TAG}" in
  A3)
    CKPT="${ROOT}/output/ablations/cirr_coarse/A3_entity_only"
    ENCODE="${COARSE_ROOT}/A3_entity_only_dual"
    RERANK="${ROOT}/output/ablations/cirr_rerank/A3_dual_k200_test1"
    ;;
  A2e)
    CKPT="${ROOT}/output/ablations/cirr_coarse/A2e_no_banzhaf_entity"
    ENCODE="${COARSE_ROOT}/A2e_no_banzhaf_entity_dual"
    RERANK="${ROOT}/output/ablations/cirr_rerank/A2e_dual_k200_test1"
    ;;
  A3s)
    CKPT="${ROOT}/output/ablations/cirr_coarse/A3s_skl03_entity"
    ENCODE="${COARSE_ROOT}/A3s_skl03_entity_dual"
    RERANK="${ROOT}/output/ablations/cirr_rerank/A3s_dual_k200_test1"
    ;;
  A3t)
    CKPT="${ROOT}/output/ablations/cirr_coarse/A3t_skl01_entity"
    ENCODE="${COARSE_ROOT}/A3t_skl01_entity_dual"
    RERANK="${ROOT}/output/ablations/cirr_rerank/A3t_dual_k200_test1"
    ;;
  A3y)
    CKPT="${ROOT}/output/ablations/cirr_coarse/A3y_freeze_orig_clip_skl03"
    ENCODE="${COARSE_ROOT}/A3y_freeze_orig_clip_skl03_dual"
    RERANK="${ROOT}/output/ablations/cirr_rerank/A3y_dual_k200_test1"
    ;;
  *)
    echo "ABORT: unknown tag ${TAG}" >&2
    exit 2
    ;;
esac

case "${ENCODE}${RERANK}" in
  */output/ablations/*) ;;
  *) echo "ABORT: paths must be under output/ablations/" >&2; exit 2 ;;
esac
case "${RERANK}" in
  */output/cirr_rerank/fusion_openclip_dual*)
    echo "ABORT: refusing mainline path" >&2; exit 2 ;;
esac

mkdir -p "${ENCODE}" "${RERANK}"
for f in "${COMPOSER}" "${STAGE1}" "${STAGE2}"; do
  [[ -f "${f}" ]] || { echo "ABORT: missing ${f}" >&2; exit 2; }
done

# SHARE_SAFE=1 (default): share a card if enough free VRAM; hard-cap our process
# so other jobs (e.g. haoyuan) keep their allocation. Empty-card still preferred.
SHARE_SAFE="${SHARE_SAFE:-1}"
RESERVE_MIB="${RESERVE_MIB:-1800}"
ENCODE_NEED_FREE="${ENCODE_NEED_FREE:-10000}"
RERANK_NEED_FREE="${RERANK_NEED_FREE:-14500}"

_gpu_free() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | awk '{print int($1)}'
}
_gpu_total() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1" | awk '{print int($1)}'
}
_gpu_used() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | awk '{print int($1)}'
}

# Sets CUDA_VISIBLE_DEVICES and MEM_FRACTION for a stage needing NEED_FREE MiB.
wait_usable_gpu() {
  local need_free="${1:?need_free}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -n "${MEM_FRACTION:-}" ]]; then
    echo "[${TAG} test1] reuse CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MEM_FRACTION=${MEM_FRACTION}"
    return 0
  fi
  while true; do
    local i free used total best=-1 best_free=-1 frac
    for i in 0 1; do
      free="$(_gpu_free "${i}")"
      used="$(_gpu_used "${i}")"
      total="$(_gpu_total "${i}")"
      # empty card: take immediately
      if [[ "${used}" -le 400 ]]; then
        best="${i}"
        best_free="${free}"
        break
      fi
      if [[ "${SHARE_SAFE}" -eq 1 && "${free}" -ge "${need_free}" && "${free}" -gt "${best_free}" ]]; then
        best="${i}"
        best_free="${free}"
      fi
    done
    if [[ "${best}" -ge 0 ]]; then
      used="$(_gpu_used "${best}")"
      total="$(_gpu_total "${best}")"
      # Cap our process so used_others + our_cap + reserve <= total
      frac="$(awk -v t="${total}" -v u="${used}" -v r="${RESERVE_MIB}" 'BEGIN{
        c=(t-u-r)/t; if(c<0.15)c=0.15; if(c>0.90)c=0.90; printf "%.2f", c
      }')"
      export CUDA_VISIBLE_DEVICES="${best}"
      export MEM_FRACTION="${frac}"
      echo "[${TAG} test1] GPU${best} free=${best_free}MiB used=${used}MiB → MEM_FRACTION=${frac} (need_free=${need_free}, reserve=${RESERVE_MIB}) $(date -Iseconds)"
      return 0
    fi
    echo "[${TAG} test1] wait GPU with free>=${need_free}MiB (share-safe, no steal) $(date -Iseconds)"
    sleep 45
  done
}

echo "[${TAG} test1] started=$(date -Iseconds) encode=${ENCODE} rerank=${RERANK}"

# 1) encode + MIC/dual submission zips
if [[ ! -f "${ENCODE}/branch_sims.npz" ]]; then
  unset MEM_FRACTION || true
  wait_usable_gpu "${ENCODE_NEED_FREE}"
  echo "[${TAG} test1] encode wti_levels=1 GPU=${CUDA_VISIBLE_DEVICES} frac=${MEM_FRACTION} $(date -Iseconds)"
  "${VENV_PY}" -u -m ablations.coarse_retrieval.eval_a3_dual_cirr_test1 \
    --ckpt "${CKPT}" \
    --out_dir "${ENCODE}" \
    --tag "${TAG}" \
    --wti_levels 1 \
    --device cuda:0 \
    --memory_fraction "${MEM_FRACTION}" \
    --coarse_root "${COARSE_ROOT}"
else
  echo "[${TAG} test1] reuse encode ${ENCODE}/branch_sims.npz"
  if [[ ! -f "${COARSE_ROOT}/${TAG}_dual_full/ablation_${TAG}_dual_full_cirr_test1.zip" ]]; then
    echo "[${TAG} test1] write MIC/dual submissions $(date -Iseconds)"
    "${VENV_PY}" -u -m ablations.coarse_retrieval.eval_a3_dual_cirr_test1 \
      --ckpt "${CKPT}" \
      --out_dir "${ENCODE}" \
      --tag "${TAG}" \
      --wti_levels 1 \
      --device cpu \
      --coarse_root "${COARSE_ROOT}"
  fi
fi

if [[ "${MODE}" == "--encode-only" || "${MODE}" == "--coarse-only" ]]; then
  echo "[${TAG} test1] ${MODE} done"
  exit 0
fi

# 2) dump Top-K (CPU)
if [[ ! -f "${RERANK}/cirr_top_${K}_test1.pt" || ! -f "${RERANK}/sim_hbi.npz" ]]; then
  echo "[${TAG} test1] dump dual Top-${K} CPU $(date -Iseconds)"
  "${VENV_PY}" -u -m ablations.coarse_retrieval.dump_cirr_ablation_dual_test1 \
    --branch_npz "${ENCODE}/branch_sims.npz" \
    --out_dir "${RERANK}" \
    --k_save "${K}" \
    --tag "${TAG}"
else
  echo "[${TAG} test1] skip dump (topk exists)"
fi

if [[ "${MODE}" == "--dump-only" ]]; then
  echo "[${TAG} test1] dump-only done"
  exit 0
fi

if [[ -f "${RERANK}/test1_fusion_meta.json" && -f "${RERANK}/ablation_${TAG}_dual_composer_cr_k200_cirr_test1.zip" ]]; then
  echo "[${TAG} test1] skip rerank (fusion zip exists)"
  exit 0
fi

# Composer/CR need more headroom; re-pick GPU and hard-cap our allocation.
unset CUDA_VISIBLE_DEVICES MEM_FRACTION || true
wait_usable_gpu "${RERANK_NEED_FREE}"
echo "[${TAG} test1] rerank GPU=${CUDA_VISIBLE_DEVICES} frac=${MEM_FRACTION} $(date -Iseconds)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

_run_fusion() {
  local only="$1"
  MEM_FRACTION="${MEM_FRACTION}" "${VENV_PY}" -u - <<PY
import os, sys, torch
frac=float(os.environ["MEM_FRACTION"])
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print(f"[mem_cap] fraction={frac}", flush=True)
sys.argv = [
  "eval_hbi_composer_cr_fusion_test1.py",
  "--topk_pt", "${RERANK}/cirr_top_${K}_test1.pt",
  "--hbi_npz", "${RERANK}/sim_hbi.npz",
  "--composer_ckpt", "${COMPOSER}",
  "--stage1_path", "${STAGE1}",
  "--stage2_path", "${STAGE2}",
  "--out_dir", "${RERANK}",
  "--k_rerank", "${K}",
  "--alpha_hbi", "0.20",
  "--beta_composer", "0.35",
  "--gamma_cr", "0.45",
  "--submission_name", "ablation_${TAG}_dual_composer_cr_k200",
  "--num_workers", "0",
  "--only", "${only}",
]
sys.path.insert(0, "${ROOT}")
import runpy
runpy.run_path("${ROOT}/HBI/cirr/vision_desc/eval_hbi_composer_cr_fusion_test1.py", run_name="__main__")
PY
}

if [[ ! -f "${RERANK}/composer_itc_scores.npz" ]]; then
  echo "[${TAG} test1] Composer ITC $(date -Iseconds)"
  _run_fusion composer
else
  echo "[${TAG} test1] skip composer"
fi

if [[ ! -f "${RERANK}/cr_nlvr_scores.npz" ]]; then
  echo "[${TAG} test1] CR NLVR $(date -Iseconds)"
  unset CUDA_VISIBLE_DEVICES MEM_FRACTION || true
  wait_usable_gpu "${RERANK_NEED_FREE}"
  _run_fusion cr
else
  echo "[${TAG} test1] skip cr"
fi

echo "[${TAG} test1] fuse + write ablation zip (not mainline submission_CIRR) $(date -Iseconds)"
"${VENV_PY}" -u -m ablations.coarse_retrieval.fuse_cirr_ablation_test1 \
  --out_dir "${RERANK}" \
  --hbi_npz "${RERANK}/sim_hbi.npz" \
  --submission_name "ablation_${TAG}_dual_composer_cr_k200" \
  --tag "${TAG}"

echo "OK: CIRR ${TAG} test1 MIC/dual + rerank K=${K} submissions written." | tee "${LOGDIR}/cirr_${TAG}_test1_done.txt"
echo "finished=$(date -Iseconds)"
echo "  MIC zip:    ${COARSE_ROOT}/${TAG}_mic_only_wti/ablation_${TAG}_mic_only_wti_cirr_test1.zip"
echo "  dual zip:   ${COARSE_ROOT}/${TAG}_dual_full/ablation_${TAG}_dual_full_cirr_test1.zip"
echo "  rerank zip: ${RERANK}/ablation_${TAG}_dual_composer_cr_k200_cirr_test1.zip"
