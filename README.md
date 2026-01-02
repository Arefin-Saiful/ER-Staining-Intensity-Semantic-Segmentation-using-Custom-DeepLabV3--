# ER-IHC Staining Intensity Semantic Segmentation (5-Class)

This repository contains a research-oriented **semantic segmentation** pipeline for **ER (Estrogen Receptor) immunohistochemistry (IHC)** intensity images.  
The model predicts **pixel-level staining intensity classes**:

- **0 — Background**
- **1 — Normal**
- **2 — Weak**
- **3 — Moderate**
- **4 — Strong**

The main contribution is a **custom DeepLabV3+ (trained from scratch)** with **deep supervision**, **class-imbalance handling**, and **robust evaluation** via **5-fold stratified cross-validation**.

---

## Key Features

- **5-class ER-IHC intensity segmentation** (Background/Normal/Weak/Moderate/Strong)
- **Custom DeepLabV3+** (scratch) + **auxiliary (deep supervision) head**
- **Class-balanced training**:
  - rare-class focused cropping / sampling
  - composite loss (**CE + Dice + Focal-Tversky**) with optional aux loss
- **Stability & performance**:
  - **EMA** (Exponential Moving Average) weights
  - **Mixed precision (AMP)**
  - warmup + cosine learning-rate schedule
  - early stopping
- **Evaluation**:
  - 5-fold CV (stratified by presence of rare classes)
  - per-class + overall metrics (IoU / Dice / Pixel Acc; with “no-background” variants)
  - confusion matrix + qualitative overlays
- **Baselines + stats**:
  - pretrained **DeepLabV3 (ResNet50)** baseline
  - pretrained **U-Net (ResNet34 encoder via segmentation_models_pytorch)** baseline
  - **Wilcoxon** paired tests + summary tables

---

## Repository Layout (recommended)

```
.
├── primary-draft-er.ipynb
├── README.md
├── DATASET_DESCRIPTION.txt
└── (outputs created at runtime)
    └── er_final_runs/
        ├── fold_0/
        ├── fold_1/
        └── ...
```

> The notebook currently uses Kaggle-style paths. You can run locally by changing the `IMG_DIR` and `MSK_DIR` variables in the config cell.

---

## Dataset (Private)

The dataset is **private** and not included in this repository.  
See **DATASET_DESCRIPTION.txt** for the expected folder structure, mask format, and class color mapping.

---

## Setup

### 1) Create an environment (example)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -U pip
pip install numpy pandas pillow matplotlib scikit-learn scipy scikit-image
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install albumentations segmentation-models-pytorch torchinfo
```

> If you’re on CPU-only, install the CPU build of PyTorch from the official instructions.

---

## How to Run

### Option A — Jupyter Notebook
1. Open `primary-draft-er.ipynb`
2. Update dataset paths:
   - `IMG_DIR = Path(".../images")`
   - `MSK_DIR = Path(".../masks")`
3. Run cells in order:
   - data pairing & summary  
   - fold creation (5-fold stratified CV)  
   - custom model training (CV)  
   - plots, qualitative results, baselines, and statistics  

### Option B — Kaggle
If you use Kaggle, keep the default folder paths and attach your dataset as an input.

---

## Mask Label Mapping (RGB → Class ID)

Your masks are expected to be RGB images where pixel colors map to class IDs:

| Class ID | Name       | RGB          |
|---------:|------------|--------------|
| 0        | Background | (0, 0, 0)    |
| 1        | Normal     | (0, 159, 255)|
| 2        | Weak       | (0, 255, 0)  |
| 3        | Moderate   | (255, 216, 0)|
| 4        | Strong     | (255, 0, 0)  |

---

## Outputs

The notebook writes results to an output directory (default example):
- fold checkpoints (best model per fold)
- training curves
- per-fold metrics and aggregated metrics (mean ± std)
- confusion matrices
- qualitative prediction overlays
- final tables + statistical tests (Wilcoxon)

---

## Notes on Reproducibility

- Fixed random seed is used for repeatability.
- For strict determinism, use deterministic flags (may reduce speed).
- Cross-validation splits are built to preserve rare class presence across folds.

---

## License

Add your preferred license (MIT/Apache-2.0/etc.) before publishing publicly.

---

## Contact

If you want, add your name, lab, and contact (email/LinkedIn) here.
