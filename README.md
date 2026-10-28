<div align="center">

# BFSAR: Banzhaf-Enhanced Fine-Grained Semantic Alignment and Refinement for Composed Image Retrieval

Anonymous code: https://anonymous.4open.science/r/BFSAR-0785

</div>

BFSAR retrieves a target image from a reference image and a modification text. Training has four stages. Inference first runs the coarse retriever, then reranks the top K candidates.

---

## Installation

Use Python **3.9** and PyTorch **2.6** (CUDA 12.4):

```bash
conda create -n BFSAR python=3.9
conda activate BFSAR
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Dataset

Prompts and generated captions are already in this repository. Download the datasets, then place them as follows.

### FashionIQ

```text
data/fashioniq/
├── images/
├── captions/
├── image_splits/
├── llm_data/          # included: target captions Tc
└── vision_desc/       # included: vision-side captions Tv
```

> Download FashionIQ from the [official repository](https://github.com/XiaoxiaoGuo/fashion-iq) and place images, captions, and image splits under `data/fashioniq/`.

### CIRR

```text
data/cirr/
├── cirr_dataset/
│   ├── train/                    # images
│   ├── dev/                      # images
│   ├── test1/                    # images
│   └── cirr/
│       ├── captions/             # cap.rc2.{train,val,test1}.json
│       └── image_splits/         # split.rc2.{train,val,test1}.json
├── llm_data/                     # included: target captions Tc
└── vision_desc/                  # included: vision-side captions Tv
```

> Download CIRR from the [official repository](https://github.com/Cuberick-Orion/CIRR) and place `train`, `dev`, `test1`, and `cirr` under `data/cirr/cirr_dataset/`.

---

## Generate Captions

```bash
python -m bfsar.caption.generate --dataset cirr
python -m bfsar.caption.generate --dataset fashioniq
```

Target captions Tc and vision-side captions Tv are already provided.

---

## Checkpoints

The download links for the trained checkpoints will be provided.

Inference also loads two public models from `weights/`, not from the Hugging Face cache:

```text
weights/blip2/blip2-itm-vit-g/    # Salesforce/blip2-itm-vit-g
weights/bert/bert-base-uncased/   # bert-base-uncased
```

---

## Training

Execute the stages in order.

### 1. Banzhaf module (teacher)

```bash
# cirr
CUDA_VISIBLE_DEVICES=<device_id> python train_bm_cirr.py

# fashioniq
CUDA_VISIBLE_DEVICES=<device_id> python train_bm_fiq.py
```

### 2. Fine-grained semantic alignment

```bash
# cirr
CUDA_VISIBLE_DEVICES=<device_id> python train_fgsa_cirr_openclip.py

# fashioniq
CUDA_VISIBLE_DEVICES=<device_id> python train_fgsa_fiq_openclip.py
```

### 3. Composed query matching

```bash
# cirr
CUDA_VISIBLE_DEVICES=<device_id> python bfsar/mvr/train_cqm_cirr.py

# fashioniq
CUDA_VISIBLE_DEVICES=<device_id> python bfsar/mvr/train_cqm_fiq.py
```

### 4. Candidate verification

```bash
# cirr
CUDA_VISIBLE_DEVICES=<device_id> bash scripts/train_cvs_cirr.sh

# fashioniq
CUDA_VISIBLE_DEVICES=<device_id> bash scripts/train_cvs_fiq.sh
```

---

## Inference

```bash
# cirr
bash scripts/eval_cirr.sh

# fashioniq
bash scripts/eval_fiq.sh
```
