# 🌾 Multimodal Paddy Yield Prediction — Kharif Season

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Sentinel--2-Satellite%20Imagery-4CAF50?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Rasterio-Geospatial-2E7D32?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
</p>

<p align="center">
  A custom dual-arm deep learning system that fuses <strong>Sentinel-2 satellite imagery</strong> with <strong>district-level agricultural metadata</strong> to predict Kharif paddy yield (Tonnes/Ha) across 26 districts of Andhra Pradesh, India.
</p>

---

## 📌 Overview

Standard yield models pick one data type — either satellite images or field statistics. This project does both.

**StanfordModel** is a custom PyTorch architecture with two parallel feature extraction arms:
- A **CNN arm** that reads 5-band `.tif` satellite images and extracts spatial crop-health patterns
- An **ANN arm** that reads tabular district metadata (rainfall, sown area, groundwater, coordinates)

Both arms produce embeddings that are concatenated and fed into a regression head to output a single prediction: **Paddy Yield in Tonnes/Hectare**.

---

## 🗂️ Repository Structure

```
multimodal-crop-yield-prediction/
│
├── data/
│   ├── Final_Model_Ready_Data.csv      # District-level tabular metadata
│   └── tif/                            # Sentinel-2 .tif images per district
│
├── model/
│   ├── stanford_model.py               # StanfordModel architecture definition
│   └── best_yield_model.pth            # Saved model weights
│
├── notebooks/
│   └── training.ipynb                  # Full training + evaluation notebook
│
├── predict.py                          # Inference engine
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
| B4 | NIR (Near-Infrared) | Crop biomass and canopy density |
| B5 | NDVI | Vegetation health index — primary yield signal |

**Preprocessing pipeline:**
- Resized to `224×224` using `F.interpolate` (handles wildly varying raw sizes)
- Normalized by dividing pixel values by `10000.0` (Sentinel-2 reflectance scale)
- `np.nan_to_num` applied to zero out curved district border artifacts

### 2. Tabular Metadata (`.csv`)

Five numerical features per district per season:

| Feature | Description |
|---------|-------------|
| `Rainfall` | Actual rainfall in mm |
| `SownArea` | Area under cultivation (Hectares) |
| `GroundWater` | District groundwater levels |
| `Latitude` | District centroid latitude |
| `Longitude` | District centroid longitude |

**Preprocessing:** Comma-stripped string parsing → `StandardScaler` normalization (fit on train set, applied to test set).

---

## 🧠 Model Architecture — `StanfordModel`

```
┌─────────────────────────────┐     ┌──────────────────────────┐
│        IMAGE ARM (CNN)       │     │    METADATA ARM (ANN)     │
│                             │     │                          │
│  Input: [B, 5, 224, 224]    │     │  Input: [B, 5]           │
│         ↓                   │     │         ↓                │
│  Conv2D(5→32) + BN + ReLU   │     │  Linear(5→64)            │
│  MaxPool2D                  │     │  BN1D + ReLU             │
│         ↓                   │     │         ↓                │
│  Conv2D(32→64) + BN + ReLU  │     │  Linear(64→128)          │
│  MaxPool2D                  │     │  BN1D + ReLU             │
│         ↓                   │     │         ↓                │
│  Conv2D(64→128) + BN + ReLU │     │  Linear(128→64)          │
│  MaxPool2D                  │     │  BN1D + ReLU             │
│         ↓                   │     │         ↓                │
│  Flatten → Linear → Dropout │     │  Embedding: [B, 64]      │
│  Embedding: [B, 256]        │     │                          │
└────────────┬────────────────┘     └─────────────┬────────────┘
             │                                    │
             └──────────── Concat ────────────────┘
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

**Training setup:**
- Loss: `MSELoss`
- Optimizer: `Adam` with learning rate scheduling
- Metric: `MAE` on held-out test set
- Best weights saved via checkpoint callback

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision rasterio pandas numpy scikit-learn matplotlib
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Running Inference

```python
from predict import predict_region_yield

# Single district prediction
result = predict_region_yield(
    region="Srikakulam",
    image_path="data/tif/Srikakulam_Kharif_2022.tif"
)

# → 🌾 PREDICTED PADDY YIELD FOR SRIKAKULAM: 2.31 Tonnes/Hectare
```

The inference engine automatically handles:
- Dynamic image resizing to `224×224`
- NaN border artifact removal
- Feature scaling using the saved `StandardScaler`

---

## ⚠️ Engineering Challenges

**1. Geospatial NaN Artifacts**
Satellite images mapped onto rectangular tensors contain `NaN` values at curved district borders. Without handling, these propagate as `nan` gradients and silently corrupt both training and inference. Fixed with `np.nan_to_num(img, nan=0.0)` as a pre-normalization defense step.

**2. Raw Image Dimensionality**
`.tif` files from different districts can have wildly different native resolutions, causing tensor shape mismatches (e.g., `[1, 5, 75556608]`). Fixed with `F.interpolate(..., size=(224, 224), mode='bilinear')` applied dynamically per sample before batching.

**3. CSV String Formatting**
Numeric fields in the raw CSV contained thousands-separator commas (e.g., `"1,23,456"`), causing silent `NaN` injection after `pd.to_numeric`. Fixed with `str.replace(',', '')` at ingestion time.

---

## 📍 Coverage

26 districts of Andhra Pradesh, India — spanning the full Kharif (monsoon) paddy season.

---

## 🤝 Acknowledgements

- Satellite data from [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) (ESA Copernicus Programme)
- Agricultural statistics from Andhra Pradesh district crop survey records
- Architecture inspired by remote sensing fusion research in agricultural AI

---

## 👤 Author

**Vinith Vanjangi** — [@Vinith-44](https://github.com/Vinith-44) | [Kaggle](https://www.kaggle.com/vinithvanjangi)

B.V. Raju Institute of Technology, Narsapur · CSE (R22) · 3rd Year
