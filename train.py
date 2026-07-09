"""
train.py — Training Script for StanfordModel
=============================================
Usage:
    python train.py

Outputs:
    best_yield_model.pth  — saved whenever validation loss improves
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StanfordDataset
from stanford_model import StanfordModel

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH      = "data/Final_Model_Ready_Data.csv"
MODEL_OUT     = "model/best_yield_model.pth"
BATCH_SIZE    = 32
EPOCHS        = 200
LEARNING_RATE = 1e-3
L2_REG        = 1e-4
ADAM_BETAS    = (0.9, 0.999)
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


def main():
    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH, thousands=",")

    train_df, temp_df   = train_test_split(df, test_size=0.2, random_state=42)
    val_df,   test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_ds = StanfordDataset(train_df)
    val_ds   = StanfordDataset(val_df,  metadata_scaler=train_ds.scaler)
    test_ds  = StanfordDataset(test_df, metadata_scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = StanfordModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           betas=ADAM_BETAS, weight_decay=L2_REG)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    os.makedirs("model", exist_ok=True)
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_losses = []
        for imgs, metas, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(imgs, metas), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_maes = [], []
        with torch.no_grad():
            for imgs, metas, labels in val_loader:
                imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device).unsqueeze(1)
                preds = model(imgs, metas)
                val_losses.append(criterion(preds, labels).item())
                val_maes.append(torch.mean(torch.abs(preds - labels)).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_mae    = np.mean(val_maes)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
