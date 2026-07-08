<div align="center">

# Multimodal Paddy Yield Prediction

### Kharif Season · Andhra Pradesh · 26 Districts

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-ESA-4CAF50?style=flat-square)](https://sentinel.esa.int)
[![Rasterio](https://img.shields.io/badge/Rasterio-Geospatial-2E7D32?style=flat-square)](https://rasterio.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br/>

> A custom **dual-arm deep learning system** that fuses **Sentinel-2 satellite imagery**
> with **district-level agricultural metadata** to predict Kharif paddy yield
> across 26 districts of Andhra Pradesh, India.

<br/>

[Overview](#-overview) · [Architecture](#-architecture) · [Dataset](#-dataset) · [Results](#-results) · [Quickstart](#-quickstart) · [Challenges](#-engineering-challenges)

</div>

---

## Overview

Most yield prediction models use either satellite images or tabular statistics — not both.

**StanfordModel** is a custom PyTorch architecture with two parallel arms that each process a different modality, then fuse their learned representations for a single regression output:

```
Sentinel-2 Image (.tif) ──► CNN Arm ──────────┐
                                               ├──► Concat ──► Regression Head ──► Yield (T/Ha)
District Metadata (.csv) ──► ANN Arm ──────────┘
```

---

## Architecture

<details>
<summary><strong>View full architecture diagram</strong></summary>

<br/>

```
┌─────────────────────────────────┐     ┌───────────────────────────┐
│         CNN ARM (Image)          │     │      ANN ARM (Metadata)    │
│                                 │     │                           │
│  Input  →  [B, 5, 224, 224]     │     │  Input  →  [B, 5]         │
│                                 │     │                           │
│  Conv2d(5→32)                   │     │  Linear(5→64)             │
│  BatchNorm2d + ReLU             │     │  BatchNorm1d + ReLU       │
│  MaxPool2d                      │     │                           │
│           ↓                     │     │  Linear(64→128)           │
│  Conv2d(32→64)                  │     │  BatchNorm1d + ReLU       │
│  BatchNorm2d + ReLU             │     │                           │
│  MaxPool2d                      │     │  Linear(128→64)           │
│           ↓                     │     │  BatchNorm1d + ReLU       │
│  Conv2d(64→128)                 │     │                           │
│  BatchNorm2d + ReLU             │     │  Embedding → [B, 64]      │
│  MaxPool2d                      │     │                           │
│           ↓                     │     └───────────┬───────────────┘
│  Flatten → Linear(? → 256)      │                 │
│  Dropout                        │                 │
│  Embedding → [B, 256]           │                 │
└──────────────┬──────────────────┘                 │
               │                                    │
               └──────────── Concatenate ───────────┘
                                  ↓
                          [B, 256 + 64] = [B, 320]
                                  ↓
                       Linear(320→128) + ReLU + Dropout
                                  ↓
                          Linear(128→32) + ReLU
                                  ↓
                             Linear(32→1)
                                  ↓
                     Predicted Yield (Tonnes/Ha)
```

</details>

| Component | Details |
|-----------|---------|
| **Image Arm** | 3× Conv2d blocks with BatchNorm + ReLU + MaxPool → flattened 256-d embedding |
| **Metadata Arm** | 3× Linear blocks with BatchNorm1d + ReLU → 64-d embedding |
| **Fusion** | Concatenation → MLP regression head → scalar output |
| **Loss** | `MSELoss` |
| **Optimizer** | `Adam` with LR scheduling |

---

## Dataset

### Sentinel-2 Satellite Imagery (`.tif`)

| Band | Channel | Purpose |
|------|---------|---------|
| B1–B3 | RGB | True-color visual context |
| B4 | Near-Infrared (NIR) | Crop biomass & canopy density |
| B5 | NDVI | Vegetation health — primary yield signal |

### District-Level Metadata (`.csv`)

| Feature | Description |
|---------|-------------|
| `Rainfall` | Actual rainfall (mm) |
| `SownArea` | Area under cultivation (Ha) |
| `GroundWater` | District groundwater levels |
| `Latitude` | District centroid latitude |
| `Longitude` | District centroid longitude |

> Preprocessing: string comma-stripping → `StandardScaler` normalization (fit on train, applied to test)

---

## Results

> Add your scatter plot here: `![Actual vs Predicted](assets/results.png)`

| Metric | Value |
|--------|-------|
| Loss | `MSELoss` on held-out test set |
| Primary Metric | `MAE` (Mean Absolute Error) |
| Coverage | 26 AP districts · Kharif season |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/Vinith-44/multimodal-crop-yield-prediction.git
cd multimodal-crop-yield-prediction
pip install torch torchvision rasterio pandas numpy scikit-learn matplotlib
```

### 2. Data layout

```
data/
├── Final_Model_Ready_Data.csv
└── tif/
    ├── Srikakulam_Kharif_2022.tif
    └── ...
model/
└── best_yield_model.pth
```

### 3. Run inference

```python
from predict import predict_region_yield

result = predict_region_yield(
    region="Srikakulam",
    image_path="data/tif/Srikakulam_Kharif_2022.tif"
)
# → 🌾 PREDICTED PADDY YIELD FOR SRIKAKULAM: 2.31 Tonnes/Hectare
```

---

## Engineering Challenges

<details>
<summary><strong>Geospatial NaN border artifacts</strong></summary>

Sentinel-2 images mapped to rectangular tensors contain `NaN` values at curved district boundaries. Without handling, these propagate as `nan` gradients and silently corrupt training.

**Fix:** `np.nan_to_num(img, nan=0.0)` applied before normalization.

</details>

<details>
<summary><strong>Raw image dimensionality mismatch</strong></summary>

`.tif` files vary wildly in native resolution across districts — causing tensor shape errors like `[1, 5, 75556608]` that crash batching.

**Fix:** `F.interpolate(..., size=(224, 224), mode='bilinear')` applied dynamically per sample.

</details>

<details>
<summary><strong>CSV string formatting errors</strong></summary>

Numeric columns contained thousands-separator commas (e.g. `"1,23,456"`), causing silent `NaN` injection after `pd.to_numeric`.

**Fix:** `str.replace(',', '')` at ingestion time before type casting.

</details>

---

## Project Structure

```
multimodal-crop-yield-prediction/
├── data/
│   ├── Final_Model_Ready_Data.csv
│   └── tif/
├── model/
│   ├── stanford_model.py
│   └── best_yield_model.pth
├── notebooks/
│   └── training.ipynb
├── predict.py
├── requirements.txt
└── README.md
```

---

<div align="center">

**Vinith Vanjangi** · [GitHub](https://github.com/Vinith-44) · [Kaggle](https://www.kaggle.com/vinithvanjangi)

B.V. Raju Institute of Technology · CSE (R22) · 3rd Year

</div>
