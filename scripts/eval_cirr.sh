#!/usr/bin/env bash
# CIRR test1: coarse retrieval, then top-200 fusion, then a CodaLab submission.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

bash "${ROOT}/scripts/eval_dpic_cirr.sh" --split test
bash "${ROOT}/scripts/eval_mvr_cirr.sh"
