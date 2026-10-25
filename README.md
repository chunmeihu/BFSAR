# BFSAR

Source for the released CIRR and FashionIQ pipeline:

1. Pretrain a frozen Banzhaf teacher on the token–region similarity matrix of positive pairs (Monte Carlo targets, MSE). The CLIP encoder stays frozen.
2. Train a single-level entity WTI retriever. A student with the same architecture is distilled from the teacher with a bidirectional KL loss. CIRR uses `skl=0.1`. FashionIQ uses `skl=1.0`.
3. Dual-path coarse retrieval (`Path_ref` and `Path_vis`) with that retriever.
4. Rerank the coarse top-K by fusing three scores: the coarse score, BLIP-2 QTR Composer, and Candidate Reranking.

Datasets, OpenCLIP weights, trained checkpoints, and the Python environment are not in this tree. Create them locally; they are gitignored.

## Environment

Python 3.9. The scripts call `.venv/bin/python` from the repository root.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt` is the rest of the environment used to run this code. It does not pin Torch; install Torch first so the CUDA 12.4 wheels are used.

## Data and weights

| Item | Default location |
|---|---|
| CIRR images | `data/cirr/img_raw` (`CIRR_IMAGE_ROOT`) |
| CIRR annotations | `data/cirr/cirr_dataset` (`CIRR_DATASET_ROOT`) |
| FashionIQ | `data/fashioniq` (`FIQ_DATA_ROOT`) |
| OpenCLIP ViT-L/14 | `weights/open_clip/hbi_format/ViT-L-14-laion2B-s32B-b82K.pt` |
| Candidate Reranking | `weights/cr/` — see `HBI/rerank/cr/DOWNLOAD.md` |
| Composer checkpoints | `output/cirr_rerank/blip2_qtr/` and `output/fiq_rerank/blip2_qtr/` |

Train Composer with `scripts/run_train_cirr_blip2_qtr_stage_a.sh` and `scripts/run_train_fiq_blip2_qtr_stage_a.sh`.

## CIRR

```bash
bash scripts/run_train_cirr_hbi_estimator_openclip.sh
bash scripts/ablations/run_cirr_a3t_skl01.sh
bash scripts/ablations/run_cirr_ablation_rerank.sh A3t
bash scripts/ablations/run_cirr_ablation_test1.sh A3t
```

`A3t` is the released CIRR setting (`wti_levels=1`, `skl=0.1`). The rerank scripts also accept other tags; those tags are not the released run. Rerank fusion weights are the frozen triple `0.20 / 0.35 / 0.45` (coarse / Composer / CR), top-200.

## FashionIQ

```bash
bash scripts/run_train_fiq_hbi_estimator_openclip.sh
bash scripts/ablations/run_fiq_coarse_train_A3.sh
bash scripts/ablations/run_fiq_a3_dual_bg.sh
bash scripts/ablations/run_fiq_a3_dual_rerank.sh
```

Released FashionIQ setting: `wti_levels=1`, `skl=1.0`. Dual-path weights are `α=0.4`, `β=0.2`, `w_ref=0.5`. Rerank fusion weights are `0.25 / 0.30 / 0.45`, top-300.

Training and evaluation write under `output/`. That directory is created at runtime and is not part of the source snapshot.
