"""
StanfordDataset — PyTorch Dataset for Multimodal Paddy Yield Prediction
========================================================================
Loads:
  - 5-band .npy tile images (pre-cropped from Sentinel-2 .tif files)
  - Tabular district metadata from Final_Model_Ready_Data.csv
  - Yield labels (Tonnes/Hectare)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


META_COLS = [
    'Actual_mm',          # Rainfall (mm)
    'GW_4_23_01',         # Groundwater level
    'RiseFall_4_23_01',   # Groundwater trend
    'Actual (Ha)',        # Sown area (hectares)
    'Deviation_percent',  # Rainfall deviation from normal
]


class StanfordDataset(Dataset):
    """
    Args:
        df               (pd.DataFrame): Rows with 'tile_path', META_COLS, 'Yield_Tonne_per_Hectare'
        metadata_scaler  (StandardScaler, optional): Pre-fit scaler. If None, fits on this dataset.
                         Always pass the train-set scaler when creating val/test datasets.
    """

    def __init__(self, df: pd.DataFrame, metadata_scaler=None):
        self.df = df.reset_index(drop=True)

        # Fit or reuse scaler
        if metadata_scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[META_COLS].fillna(0))
        else:
            self.scaler = metadata_scaler

        self.meta_data = self.scaler.transform(
            self.df[META_COLS].fillna(0).values
        ).astype('float32')

        self.paths  = self.df['tile_path'].values
        self.yields = self.df['Yield_Tonne_per_Hectare'].values.astype('float32')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load pre-cropped .npy tile — shape [5, 224, 224]
        try:
            img = np.load(self.paths[idx]).astype('float32')
            img[:4] /= 10000.0  # Normalize RGB+NIR from DN range (0-10000) to (0-1)
            # Band 5 (NDVI) is already in [-1, 1], leave unchanged
        except Exception:
            img = np.zeros((5, 224, 224), dtype='float32')

        return (
            torch.from_numpy(img),
            torch.tensor(self.meta_data[idx]),
            torch.tensor(self.yields[idx]),
        )
