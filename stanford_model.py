"""
StanfordModel — Dual-Arm Multimodal CNN for Paddy Yield Prediction
====================================================================
Architecture:
  - CNN Arm   : Processes 5-band Sentinel-2 satellite images [B, 5, 224, 224]
  - Meta Arm  : Processes district-level tabular metadata   [B, 5]
  - Head      : Concatenates both embeddings → regression output (Tonnes/Ha)
"""

import torch
import torch.nn as nn


class StanfordModel(nn.Module):
    """
    Dual-arm multimodal regression model for crop yield prediction.

    Input:
        img  (Tensor): Shape [B, 5, 224, 224] — 5-band Sentinel-2 image
        meta (Tensor): Shape [B, 5]           — scaled district metadata

    Output:
        Tensor of shape [B, 1] — predicted yield in Tonnes/Hectare
    """

    def __init__(self):
        super(StanfordModel, self).__init__()

        # ── CNN ARM ──────────────────────────────────────────────────────────
        # 3 Conv blocks: Conv2d → BatchNorm → ReLU → MaxPool
        # Input:  [B, 5, 224, 224]
        # Output: [B, 128, 28, 28] → flattened → 64-d embedding
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool        = nn.MaxPool2d(2)
        self.relu        = nn.ReLU()
        self.flatten     = nn.Flatten()
        self.fc_cnn      = nn.Linear(128 * 28 * 28, 64)
        self.dropout_cnn = nn.Dropout(0.5)

        # ── METADATA ARM ─────────────────────────────────────────────────────
        # 2 Linear blocks: Linear → BatchNorm → ReLU
        # Input:  [B, 5]
        # Output: 8-d embedding
        self.fc_meta1 = nn.Linear(5, 16)
        self.bn_meta  = nn.BatchNorm1d(16)
        self.fc_meta2 = nn.Linear(16, 8)

        # ── REGRESSION HEAD ──────────────────────────────────────────────────
        # Input:  [B, 64 + 8] = [B, 72]
        # Output: [B, 1]  —  Yield (Tonnes/Ha)
        self.fc_final1    = nn.Linear(64 + 8, 64)
        self.dropout_head = nn.Dropout(0.3)
        self.fc_final2    = nn.Linear(64, 32)
        self.fc_out       = nn.Linear(32, 1)

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # CNN arm
        x = self.pool(self.relu(self.bn1(self.conv1(img))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout_cnn(self.relu(self.fc_cnn(x)))

        # Metadata arm
        y = self.relu(self.bn_meta(self.fc_meta1(meta)))
        y = self.relu(self.fc_meta2(y))

        # Fusion + head
        z = torch.cat((x, y), dim=1)
        z = self.dropout_head(self.relu(self.fc_final1(z)))
        z = self.relu(self.fc_final2(z))
        return self.fc_out(z)
