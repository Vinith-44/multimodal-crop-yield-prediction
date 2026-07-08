<div align="center">

# 🌾 Multimodal Paddy Yield Prediction

### Kharif Season · Andhra Pradesh · 26 Districts

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-ESA%20Copernicus-4CAF50?style=flat-square)](https://sentinel.esa.int)
[![Rasterio](https://img.shields.io/badge/Rasterio-Geospatial-2E7D32?style=flat-square)](https://rasterio.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<br/>

> A custom **dual-arm deep learning system** that fuses **Sentinel-2 satellite imagery**  
> with **district-level agricultural metadata** to predict Kharif paddy yield (Tonnes/Ha)  
> across **26 districts of Andhra Pradesh, India.**

<br/>

[Overview](#-overview) · [Architecture](#-model-architecture--stanfordmodel) · [Dataset](#-datasets) · [Quickstart](#-quickstart) · [Challenges](#️-engineering-challenges) · [Acknowledgements](#-acknowledgements)

</div>

---

## 📌 Overview

Standard yield models pick one data type — either satellite images or field statistics. **This project uses both.**

**StanfordModel** is a custom PyTorch architecture with two parallel feature extraction arms:

- A **CNN arm** that reads 5-band `.tif` satellite images and extracts spatial crop-health patterns
- An **ANN arm** that reads tabular district metadata (rainfall, sown area, groundwater, coordinates)

Both arms produce embeddings that are concatenated and fed into a regression head to output a single prediction: **Paddy Yield in Tonnes/Hectare.**

---

## 🗂️ Repository Structure

```
multimodal-crop-yield-prediction/
│
├── data/
│   ├── Final_Model_Ready_Data.csv       # District-level tabular metadata
│   └── tif/                             # Sentinel-2 .tif images per district
│
├── model/
│   ├── stanford_model.py                # StanfordModel architecture definition
│   └── best_yield_model.pth             # Saved best model weights
│
├── notebooks/
│   └── training.ipynb                   # Full training + evaluation notebook
│
├── predict.py                           # Inference engine
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

### 1. Satellite Imagery — Sentinel-2 (`.tif`)

Each district has a corresponding multi-band GeoTIFF file capturing Kharif season vegetation:

| Band | Channel | Role |
|------|---------|------|
| B1–B3 | RGB (Red, Green, Blue) | True-color visual context |
| B4 | Near-Infrared (NIR) | Crop biomass and canopy density |
| B5 | NDVI | Vegetation health index — primary yield signal |

**Preprocessing pipeline:**
- Resized to `224×224` using `F.interpolate` (handles wildly varying raw sizes across districts)
- Normalized by dividing pixel values by `10000.0` (Sentinel-2 standard reflectance scale)
- `np.nan_to_num` applied to zero out curved district border artifacts before normalization

### 2. District-Level Metadata (`.csv`)

Five numerical features per district per season:

| Feature | Description |
|---------|-------------|
| `Rainfall` | Actual rainfall (mm) |
| `SownArea` | Area under cultivation (Hectares) |
| `GroundWater` | District groundwater levels |
| `Latitude` | District centroid latitude |
| `Longitude` | District centroid longitude |

**Preprocessing:** Comma-stripped string parsing → `StandardScaler` normalization (fit on train set, applied to test set)

---

## 🧠 Model Architecture — `StanfordModel`

```
┌──────────────────────────────────┐     ┌──────────────────────────────┐
│          CNN ARM (Image)          │     │       ANN ARM (Metadata)      │
│                                  │     │                              │
│   Input: [B, 5, 224, 224]        │     │   Input: [B, 5]              │
│            ↓                     │     │            ↓                 │
│   Conv2d(5→32) + BN2d + ReLU     │     │   Linear(5→64)               │
│   MaxPool2d                      │     │   BatchNorm1d + ReLU         │
│            ↓                     │     │            ↓                 │
│   Conv2d(32→64) + BN2d + ReLU    │     │   Linear(64→128)             │
│   MaxPool2d                      │     │   BatchNorm1d + ReLU         │
│            ↓                     │     │            ↓                 │
│   Conv2d(64→128) + BN2d + ReLU   │     │   Linear(128→64)             │
│   MaxPool2d                      │     │   BatchNorm1d + ReLU         │
│            ↓                     │     │            ↓                 │
│   Flatten → Linear → Dropout     │     │   Embedding: [B, 64]         │
│   Embedding: [B, 256]            │     │                              │
└───────────────┬──────────────────┘     └──────────────┬───────────────┘
                │                                       │
                └──────────────── Concat ───────────────┘
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

**Training configuration:**

| Setting | Value |
|---------|-------|
| Loss | `MSELoss` |
| Optimizer | `Adam` with LR scheduling |
| Eval Metric | `MAE` on held-out test set |
| Checkpointing | Best weights saved via loss callback |

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Vinith-44/multimodal-crop-yield-prediction.git
cd multimodal-crop-yield-prediction
```

### 2. Install dependencies

```bash
pip install torch torchvision rasterio pandas numpy scikit-learn matplotlib
```

Or via requirements file:

```bash
pip install -r requirements.txt
```

### 3. Prepare your data

- Place `Final_Model_Ready_Data.csv` in the `data/` directory
- Place `.tif` satellite images in `data/tif/`
- Place `best_yield_model.pth` in the `model/` directory

### 4. Run inference

```python
from predict import predict_region_yield

result = predict_region_yield(
    region="Srikakulam",
    image_path="data/tif/Srikakulam_Kharif_2022.tif"
)
# → 🌾 PREDICTED PADDY YIELD FOR SRIKAKULAM: 2.31 Tonnes/Hectare
```

The inference engine automatically handles resizing, NaN removal, and feature scaling.

---

## ⚠️ Engineering Challenges

### 1. Geospatial NaN Border Artifacts

Sentinel-2 images mapped onto rectangular tensors contain `NaN` values at curved district boundaries. Without handling, these silently propagate as `nan` gradients and corrupt both training and inference.

**Fix:** `np.nan_to_num(img, nan=0.0)` applied as a mandatory pre-normalization step.

### 2. Raw Image Dimensionality Mismatch

`.tif` files from different districts have wildly different native resolutions, causing tensor shape mismatches at batching time (e.g., `[1, 5, 75556608]` vs expected `[1, 5, 224, 224]`).

**Fix:** `F.interpolate(..., size=(224, 224), mode='bilinear')` applied dynamically per sample before collation.

### 3. CSV String Formatting Errors

Numeric columns in the raw dataset contained thousands-separator commas (e.g., `"1,23,456"`), causing silent `NaN` injection after `pd.to_numeric`.

**Fix:** `str.replace(',', '')` applied at ingestion time, before type casting.

---

## 📍 Coverage

26 districts of Andhra Pradesh, India — spanning the full Kharif (monsoon) paddy growing season.

---

## 🤝 Acknowledgements

- Satellite imagery from [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) (ESA Copernicus Programme)
- Agricultural statistics from Andhra Pradesh district crop survey records
- Architecture inspired by remote sensing fusion research in agricultural AI

---

<div align="center">

**Vinith Vanjangi** · [GitHub @Vinith-44](https://github.com/Vinith-44) · [Kaggle](https://www.kaggle.com/vinithvanjangi)

*B.V. Raju Institute of Technology · CSE (R22) · 3rd Year*

</div>
