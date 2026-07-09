"""
StanfordDataset — PyTorch Dataset for Multimodal Paddy Yield Prediction
========================================================================
Loads 5-band .npy tile images + tabular district metadata from
Final_Model_Ready_Data.csv and returns (image, metadata, yield) tuples.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


META_COLS = ['Rainfall', 'SownArea', 'GroundWater', 'Latitude', 'Longitude']


class StanfordDataset(Dataset):
    """
    Args:
        df               : DataFrame with columns: tile_path, META_COLS, Yield_Tonne_per_Hectare
        metadata_scaler  : Pre-fit StandardScaler. If None, fits on this dataset.
                           Always pass the train scaler when creating val/test sets.
    """

    def __init__(self, df: pd.DataFrame, metadata_scaler=None):
        self.df = df.reset_index(drop=True)

        # Force numeric — handles stray comma-formatted strings
        for col in META_COLS:
            if self.df[col].dtype == object:
                self.df[col] = self.df[col].astype(str).str.replace(",", "").astype(float)

        if metadata_scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[META_COLS].fillna(0))
        else:
            self.scaler = metadata_scaler

        self.meta_data = self.scaler.transform(
            self.df[META_COLS].fillna(0).values
        ).astype("float32")

        self.paths  = self.df["tile_path"].values
        self.yields = self.df["Yield_Tonne_per_Hectare"].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img = np.load(self.paths[idx]).astype("float32")  # [5, 224, 224]
            img[:4] /= 10000.0   # Normalize RGB+NIR; NDVI (band 5) already in [-1, 1]
        except Exception:
            img = np.zeros((5, 224, 224), dtype="float32")

        return (
            torch.from_numpy(img),
            torch.tensor(self.meta_data[idx]),
            torch.tensor(self.yields[idx]),
        )
