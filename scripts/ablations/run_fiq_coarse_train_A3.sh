#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export ABLATION_ID=A3
export OUT_DIR="${ROOT}/output/ablations/fiq_coarse/A3_entity_only"
export ESTIMATOR="${ROOT}/output/fiq_estimator_openclip/banzhaf_estimator.epoch10.pt"
export SKL=1
export WTI_LEVELS=1
export MASTER_PORT="${MASTER_PORT:-29653}"
bash "${ROOT}/scripts/ablations/run_fiq_coarse_train.sh" "$@"
