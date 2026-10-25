# Rerank backends (Composer + Candidate-Reranking)

After coarse HBI retrieval, GBCIR uses two sibling rerankers:

| Package | Role | Code | Weights |
|---------|------|------|---------|
| `HBI/rerank/composer` | BLIP-2 QTR Composer (ITC) | `HBI/{cirr,fiq}/vision_desc/*blip2_qtr*` | `output/{cirr,fiq}_rerank/blip2_qtr/` |
| `HBI/rerank/cr` | Candidate-Reranking BLIP Stage-1/2 | vendored `HBI/rerank/cr/src` | `weights/cr/` |

## CR checkpoints

Place official CR checkpoints under `weights/cr/` (same layout as upstream `models/`):

```
weights/cr/
  stage1/CIRR/saved_models/blip_mean.pt
  stage2/CIRR/saved_models/blip_mean.pt
  stage1/fashionIQ/saved_models/blip.pt
  stage2/fashionIQ/saved_models/blip.pt
  HBI_coarse/cirr_top_200_{val,test1}.pt   # optional; HBI coarse Top-K
```

Download: see `HBI/rerank/cr/DOWNLOAD.md` (OneDrive from the CR authors).

Train FIQ CR from scratch (optional; SOTA uses the released ckpts above):

```bash
bash scripts/run_train_fiq_cr_stage1.sh
TOPK_PT=weights/cr/stage1/fashionIQ/fiq_top_200_val_dress.pt bash scripts/run_train_fiq_cr_stage2.sh
```

Data roots: `FIQ_DATA_ROOT` → `data/fashioniq` (png/jpg). Train dumps → `output/cr_train/`.

Override roots with `GBCIR_CR_ROOT` / `GBCIR_CR_WEIGHTS` / `GBCIR_CR_OUTPUT` if needed.

## Quick smoke test

```bash
cd GBCIR-main
.venv/bin/python -c "
from pathlib import Path
from HBI.cirr.vision_desc.hbi_blip_rerank import load_blip_models, DEFAULT_STAGE1, DEFAULT_STAGE2
m1, m2, prep, _ = load_blip_models(DEFAULT_STAGE1, DEFAULT_STAGE2)
print('ok', DEFAULT_STAGE1.name, DEFAULT_STAGE2.name)
"
```

Attribution: Candidate-Reranking code is MIT-licensed; see `HBI/rerank/cr/LICENSE`.
