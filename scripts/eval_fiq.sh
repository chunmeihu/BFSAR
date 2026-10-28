#!/usr/bin/env bash
# FashionIQ validation: coarse retrieval, then top-200 fusion on the large and small galleries.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

bash "${ROOT}/scripts/eval_dpic_fiq.sh" --split val
bash "${ROOT}/scripts/eval_mvr_fiq.sh"
