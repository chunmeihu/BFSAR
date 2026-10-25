#!/usr/bin/env bash
# CIRR ablation: dual coarse Top-200 dump (CPU) + Composer ITC + CR NLVR + fuse.
# Writes only under output/ablations/cirr_rerank/. Headline fuse is frozen
# mainline 0.20/0.35/0.45 (not retuned). Does not touch mainline ckpts.
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

TAG="${1:?usage: run_cirr_ablation_rerank.sh A3|A2e|A3t [--dump-only]}"
DUMP_ONLY=0
if [[ "${2:-}" == "--dump-only" ]]; then
  DUMP_ONLY=1
fi

K=200
COMPOSER="${ROOT}/output/cirr_rerank/blip2_qtr/qtr_v1_stageA_g32/qtr_stagea_best_pool_r1.pt"
STAGE1="${ROOT}/weights/cr/stage1/CIRR/saved_models/blip_mean.pt"
STAGE2="${ROOT}/weights/cr/stage2/CIRR/saved_models/blip_mean.pt"
LOGDIR="${ROOT}/output/ablations/logs"
mkdir -p "${LOGDIR}"

case "${TAG}" in
  A3)
    DUAL="${ROOT}/output/ablations/cirr_coarse/A3_entity_only_dual"
    OUT="${ROOT}/output/ablations/cirr_rerank/A3_dual_k200"
    ;;
  A2e)
    DUAL="${ROOT}/output/ablations/cirr_coarse/A2e_no_banzhaf_entity_dual"
    OUT="${ROOT}/output/ablations/cirr_rerank/A2e_dual_k200"
    ;;
  A3t)
    DUAL="${ROOT}/output/ablations/cirr_coarse/A3t_skl01_entity_dual"
    OUT="${ROOT}/output/ablations/cirr_rerank/A3t_dual_k200"
    ;;
  *)
    echo "ABORT: unknown tag ${TAG}" >&2
    exit 2
    ;;
esac

case "${OUT}" in
  */output/ablations/cirr_rerank/*) ;;
  *) echo "ABORT: OUT must be under output/ablations/cirr_rerank/: ${OUT}" >&2; exit 2 ;;
esac
case "${OUT}" in
  */output/cirr_rerank/fusion_openclip_dual*)
    echo "ABORT: refusing mainline path" >&2; exit 2 ;;
esac

mkdir -p "${OUT}"
BRANCH="${DUAL}/branch_sims.npz"
[[ -f "${BRANCH}" ]] || { echo "ABORT: missing ${BRANCH}" >&2; exit 2; }
for f in "${COMPOSER}" "${STAGE1}" "${STAGE2}"; do
  [[ -f "${f}" ]] || { echo "ABORT: missing ${f}" >&2; exit 2; }
done

wait_free_gpu() {
  local need="${1:-18000}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[${TAG} rerank] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} already set"
    return 0
  fi
  while true; do
    local i used nproc
    for i in 0 1; do
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${i}" | awk '{print int($1)}')"
      nproc="$(nvidia-smi -i "${i}" --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]' || true)"
      if [[ "${used}" -le 400 && "${nproc}" -eq 0 ]]; then
        echo "[${TAG} rerank] GPU${i} empty (used=${used}MiB); take this card"
        export CUDA_VISIBLE_DEVICES="${i}"
        return 0
      fi
    done
    echo "[${TAG} rerank] wait empty GPU (>=${need} free, used<=400, no apps) $(date -Iseconds)"
    sleep 60
  done
}

if [[ ! -f "${OUT}/cirr_top_${K}_val.pt" || ! -f "${OUT}/sim_hbi.npz" ]]; then
  echo "[${TAG}] dump dual Top-${K} CPU $(date -Iseconds)"
  "${VENV_PY}" -u -m ablations.coarse_retrieval.dump_cirr_ablation_dual_topk \
    --branch_npz "${BRANCH}" \
    --out_dir "${OUT}" \
    --k_save "${K}" \
    --tag "${TAG}"
else
  echo "[${TAG}] skip dump (topk exists)"
fi

if [[ "${DUMP_ONLY}" -eq 1 ]]; then
  echo "[${TAG}] dump-only done"
  exit 0
fi

if [[ -f "${OUT}/rerank_final.json" ]]; then
  echo "[${TAG}] skip rerank (rerank_final.json exists)"
  exit 0
fi

wait_free_gpu 18000
echo "[${TAG}] GPU=${CUDA_VISIBLE_DEVICES} $(date -Iseconds) out=${OUT}"

COMMON=(
  -u "${ROOT}/HBI/cirr/vision_desc/eval_hbi_composer_cr_fusion.py"
  --topk_pt "${OUT}/cirr_top_${K}_val.pt"
  --hbi_npz "${OUT}/sim_hbi.npz"
  --composer_ckpt "${COMPOSER}"
  --stage1_path "${STAGE1}"
  --stage2_path "${STAGE2}"
  --out_dir "${OUT}"
  --k_rerank "${K}"
  --num_workers 0
)

if [[ ! -f "${OUT}/composer_itc_scores.npz" ]]; then
  echo "[${TAG}] Composer ITC $(date -Iseconds)"
  "${VENV_PY}" "${COMMON[@]}" --only composer
else
  echo "[${TAG}] skip composer"
fi

if [[ ! -f "${OUT}/cr_nlvr_scores.npz" ]]; then
  echo "[${TAG}] CR NLVR $(date -Iseconds)"
  "${VENV_PY}" "${COMMON[@]}" --only cr
else
  echo "[${TAG}] skip cr"
fi

if [[ ! -f "${OUT}/fusion_metrics.json" ]]; then
  echo "[${TAG}] fuse grid $(date -Iseconds)"
  "${VENV_PY}" "${COMMON[@]}" --only fuse
else
  echo "[${TAG}] skip fuse grid"
fi

echo "[${TAG}] frozen-mainline report $(date -Iseconds)"
"${VENV_PY}" -u -m ablations.coarse_retrieval.report_cirr_ablation_rerank \
  --out_dir "${OUT}" \
  --dual_metrics "${DUAL}/metrics.json" \
  --tag "${TAG}"

echo "OK: CIRR ${TAG} dual+rerank K=${K} done." | tee "${LOGDIR}/cirr_${TAG}_rerank_done.txt"
echo "finished=$(date -Iseconds)"
