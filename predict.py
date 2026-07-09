"""
predict.py — Inference Engine for StanfordModel
================================================
Loads a trained StanfordModel and predicts paddy yield for a given district.

Usage (Python):
    from predict import predict_yield
    result = predict_yield(district="Srikakulam", tif_path="data/tif/Srikakulam_Kharif_2022.tif")

Usage (CLI):
    python predict.py --district Srikakulam --tif data/tif/Srikakulam_Kharif_2022.tif
"""

import argparse

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from stanford_model import StanfordModel

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model/best_yield_model.pth"
CSV_PATH   = "data/Final_Model_Ready_Data.csv"
IMG_SIZE   = 224
META_COLS  = [
    'Actual_mm',
    'GW_4_23_01',
    'RiseFall_4_23_01',
    'Actual (Ha)',
    'Deviation_percent',
]
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> StanfordModel:
    model = StanfordModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def _load_image(tif_path: str) -> torch.Tensor:
    """Read a 5-band .tif, resize to 224×224, normalize, return [1, 5, 224, 224]."""
    with rasterio.open(tif_path) as src:
        img = src.read().astype("float32")  # [5, H, W]

    img = np.nan_to_num(img, nan=0.0)       # Zero out border NaN artifacts
    img /= 10000.0                           # DN → reflectance (bands 1-4)

    tensor = torch.from_numpy(img).unsqueeze(0)  # [1, 5, H, W]
    tensor = F.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return tensor.to(device)


def _load_metadata(district: str, csv_path: str = CSV_PATH) -> torch.Tensor:
    """Fetch district row from CSV, scale, return [1, 5] tensor."""
    df = pd.read_csv(csv_path, thousands=",")

    # Try common district column names
    for col in ["District", "District_Name", "Dist Name", "district_name"]:
        if col in df.columns:
            row = df[df[col].str.strip().str.title() == district.strip().title()]
            if not row.empty:
                break
    else:
        raise ValueError(f"District '{district}' not found in {csv_path}.")

    scaler = StandardScaler()
    scaler.fit(df[META_COLS].fillna(0))

    meta = scaler.transform(row[META_COLS].fillna(0).values).astype("float32")
    return torch.from_numpy(meta).to(device)


def predict_yield(district: str, tif_path: str) -> float:
    """
    Predict paddy yield for a district.

    Args:
        district  : District name (e.g. "Srikakulam")
        tif_path  : Path to the district's Sentinel-2 .tif image

    Returns:
        Predicted yield in Tonnes/Hectare (float)
    """
    model  = _load_model()
    img    = _load_image(tif_path)
    meta   = _load_metadata(district)

    with torch.no_grad():
        pred = model(img, meta).item()

    print(f"\n🌾 PREDICTED PADDY YIELD FOR {district.upper()}: {pred:.4f} Tonnes/Hectare\n")
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StanfordModel Inference")
    parser.add_argument("--district", required=True, help="District name (e.g. Srikakulam)")
    parser.add_argument("--tif",      required=True, help="Path to the district .tif image")
    args = parser.parse_args()
    predict_yield(args.district, args.tif)
