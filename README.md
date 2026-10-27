# BFSAR

Banzhaf-Enhanced Fine-Grained Semantic Alignment and Refinement for Composed Image Retrieval.

Anonymous code and paper materials: https://anonymous.4open.science/r/BFSAR-0785

BFSAR retrieves a target image from a reference image and a modification sentence. The released pipeline has three parts:

1. **FGSA** — fine-grained semantic alignment. Token and region weights come from two separate MLPs. A frozen Banzhaf teacher, trained with MSE against Monte Carlo Banzhaf values, distills a student with a bidirectional KL loss. Both Banzhaf modules are dropped at inference.
2. **DPIC** — dual-path intent calibration. **RASD** mixes the reference image into the coarse score. **DASD** mixes the vision-side caption. The coarse score is `w * S_a + (1 - w) * S_b`.
3. **MVR** — multi-view reranking on the top-200 candidates. The fused score is `λ1 * S_CRS + λ2 * S_CQM + λ3 * S_CVS`. **CQM** is composed query matching (cosine, contrastive). **CVS** is candidate verification scoring (an MLP head).

## Requirements

Tested with Python 3.9 and PyTorch 2.6 (CUDA 12.4).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`comet_ml` is optional. It is only used if you turn on remote logging while training CVS.

## Dataset

Download [CIRR](https://github.com/Cuberick-Orion/CIRR) and [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq) and place the images as follows:

```text
data/cirr/img_raw/
data/cirr/cirr_dataset/
data/fashioniq/images/
data/fashioniq/captions/
data/fashioniq/image_splits/
```

Generated text is already in this repository:

| Path | Content |
| --- | --- |
| `data/prompts/target_caption.txt` | Prompt for the target caption `Tc`. This is the MCoT four-step prompt. The JSON field is `ModifiedImageCaption`. |
| `data/prompts/vision_caption.txt` | Prompt for the vision-side caption `Tv`. The JSON field is `target_query`. |
| `data/cirr/llm_data/` | CIRR target captions |
| `data/fashioniq/llm_data/` | FashionIQ target captions |
| `data/cirr/vision_desc/` | CIRR reference, target, and vision-side captions |
| `data/fashioniq/vision_desc/` | FashionIQ reference, target, and vision-side captions |

## Weights

Checkpoints are too large for git. Download them from Baidu Netdisk and keep the relative paths below.

Download: `https://pan.baidu.com/s/REPLACE_ME` (extraction code: `REPLACE_ME`)

```text
weights/openclip/ViT-L-14-laion2B-s32B-b82K.pt
weights/bm/cirr/banzhaf_module.pt
weights/bm/fashioniq/banzhaf_module.pt
weights/fgsa/cirr/pytorch_model.bin
weights/fgsa/fashioniq/pytorch_model.bin
weights/cqm/cirr/cqm.pt
weights/cqm/fashioniq/cqm.pt
weights/cvs/cirr/stage1.pt
weights/cvs/cirr/stage2.pt
weights/cvs/fashioniq/stage1.pt
weights/cvs/fashioniq/stage2.pt
```

`stage2.pt` is the CVS checkpoint used for the reported results. `stage1.pt` is the BLIP retrieval checkpoint it loads. CQM also needs the BLIP-2 ITM initialization (`Salesforce/blip2-itm-vit-g`) in the Hugging Face cache.

## Quick Start

### 1. Banzhaf teacher

```bash
bash scripts/train_bm_cirr.sh
bash scripts/train_bm_fiq.sh
```

10 epochs, learning rate `1e-2`, 12 crops, 50 Monte Carlo samples. Copy `banzhaf_estimator.epoch10.pt` to the `weights/bm/...` path above.

### 2. Fine-grained semantic alignment

```bash
bash scripts/train_fgsa_cirr.sh
bash scripts/train_fgsa_fiq.sh
```

Important parameters:

1. One matching level (`--fgsa_levels 1`).
2. 10 epochs, BertAdam, learning rate `1e-4`.
3. `λB = 0.1` on CIRR and `λB = 1.0` on FashionIQ.
4. Per-GPU batch size 16 on CIRR and 64 on FashionIQ.

### 3. Dual-path intent calibration

```bash
bash scripts/eval_dpic_cirr.sh
bash scripts/eval_dpic_fiq.sh
```

| | RASD `(α, β)` | DASD `(α, β)` | `w` |
| --- | --- | --- | --- |
| CIRR | 0.5, 0.2 | 0.2, 0.8 | 0.2 |
| FashionIQ | 0.4, 0.2 | 0.4, 0.2 | 0.5 |

### 4. Composed query matching

```bash
bash scripts/train_cqm_cirr.sh
bash scripts/train_cqm_fiq.sh
```

Contrastive loss only. Temperature `0.07`, learning rate `2e-5`, 40 epochs, per-GPU batch size 16.

### 5. Candidate verification and fusion

CVS training code is `bfsar/mvr/cvs/src/stage2_train.py`. `scripts/train_cvs.sh` launches it. The reported table uses the released `stage2.pt` files, not a fresh run of that script.

```bash
bash scripts/eval_mvr_cirr.sh
bash scripts/eval_mvr_fiq.sh
```

Both datasets rerank the top 200 candidates.

| | `λ1` coarse | `λ2` CQM | `λ3` CVS |
| --- | --- | --- | --- |
| CIRR | 0.25 | 0.40 | 0.35 |
| FashionIQ | 0.15 | 0.40 | 0.45 |

## Layout

```text
bfsar/models/     FGSA matcher and the Banzhaf module
bfsar/cirr/       CIRR data and the retrieval model
bfsar/fiq/        FashionIQ data
bfsar/dpic/       RASD / DASD calibration
bfsar/mvr/        CQM, CVS, and the λ1/λ2/λ3 fusion
bfsar/mvr/cvs/    CVS training code
data/             prompts and generated captions
scripts/          train and eval entry points
weights/          checkpoints from the link above (not in git)
```
