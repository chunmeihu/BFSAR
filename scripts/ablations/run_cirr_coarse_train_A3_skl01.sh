#!/usr/bin/env bash
# CIRR A3t: Banzhaf entity-only, skl=0.1 (same 2×16 as A3/A3s). New OUT_DIR only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export ABLATION_ID=A3t
export OUT_DIR="${ROOT}/output/ablations/cirr_coarse/A3t_skl01_entity"
export ESTIMATOR="${ROOT}/output/cirr_estimator_openclip/banzhaf_estimator.epoch10.pt"
export SKL=0.1
export WTI_LEVELS=1
export MASTER_PORT="${MASTER_PORT:-29666}"
bash "${ROOT}/scripts/ablations/run_cirr_coarse_train.sh" "$@"
