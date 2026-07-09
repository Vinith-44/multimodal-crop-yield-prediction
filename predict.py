"""
predict.py — Inference Engine for StanfordModel
================================================
Usage (Python):
    from predict import predict_region_yield
    result = predict_region_yield("Srikakulam", "data/tif/Srikakulam_Kharif_2022.tif")

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
META_COLS  = ['Rainfall', 'SownArea', 'GroundWater', 'Latitude', 'Longitude']
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
        img = src.read().astype("float32")          # [5, H, W]

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)  # zero NaN border artifacts
    if img.shape[0] >= 4:
        img[:4] /= 10000.0                          # DN → reflectance for RGB+NIR

    tensor = torch.from_numpy(img).unsqueeze(0)     # [1, 5, H, W]
    tensor = F.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return tensor.to(device)


def _load_metadata(district: str) -> torch.Tensor:
    """Fetch district row from CSV, scale metadata, return [1, 5] tensor."""
    df = pd.read_csv(CSV_PATH, thousands=",")

    # Force numeric in case any commas slipped through
    for col in META_COLS:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)

    scaler = StandardScaler()
    scaler.fit(df[META_COLS].fillna(0))

    row = df[df["District"].str.contains(district, case=False, na=False)]
    if row.empty:
        raise ValueError(f"District '{district}' not found in {CSV_PATH}.")

    meta = scaler.transform(row.iloc[[0]][META_COLS].astype(float).fillna(0))
    meta = np.nan_to_num(meta, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return torch.from_numpy(meta).to(device)


def predict_region_yield(region_name: str, tif_image_path: str) -> str:
    """
    Predict paddy yield for a district.

    Args:
        region_name     : District name (e.g. "Srikakulam")
        tif_image_path  : Path to the district's Sentinel-2 .tif image

    Returns:
        Result string with predicted yield in Tonnes/Hectare
    """
    print(f"\n🌾 Analyzing data for region: {region_name}...")
    model  = _load_model()
    img    = _load_image(tif_image_path)
    meta   = _load_metadata(region_name)

    with torch.no_grad():
        prediction = model(img, meta).item()

    result = f"🚀 PREDICTED PADDY YIELD FOR {region_name.upper()}: {prediction:.4f} Tonnes/Hectare"
    print("-" * 65)
    print(result)
    print("-" * 65)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StanfordModel Inference")
    parser.add_argument("--district", required=True, help="District name e.g. Srikakulam")
    parser.add_argument("--tif",      required=True, help="Path to district .tif image")
    args = parser.parse_args()
    predict_region_yield(args.district, args.tif)
