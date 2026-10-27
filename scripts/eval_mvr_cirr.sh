#!/usr/bin/env bash
# CIRR multi-view reranking. K = 200, lambda = (0.25, 0.40, 0.35).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then PY="${ROOT}/.venv/bin/python"; else PY="${PYTHON:-python}"; fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "${PY}" -m bfsar.mvr.fuse_cirr \
  --k_rerank 200 \
  --lambda_1 0.25 \
  --lambda_2 0.40 \
  --lambda_3 0.35 \
  --cqm_ckpt "${CQM:-${ROOT}/weights/cqm/cirr/cqm.pt}" \
  --stage1_path "${CVS1:-${ROOT}/weights/cvs/cirr/stage1.pt}" \
  --stage2_path "${CVS2:-${ROOT}/weights/cvs/cirr/stage2.pt}" \
  --out_dir "${OUT:-${ROOT}/output/mvr/cirr}" \
  "$@"
