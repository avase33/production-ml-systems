"""
LSTM Failure Prediction Model for Oilfield Equipment
=====================================================
Predicts equipment failure 48 hours in advance using time-series sensor data.
Trained on high-frequency IoT sensor streams from 200+ active wells.

Sensors monitored:
  - Wellhead pressure (psi)
  - Pump inlet/outlet temperature (°F)
  - Vibration (mm/s RMS)
  - Flow rate (bbl/day)
  - Motor current draw (amps)
  - Rotational speed (RPM)

Author: Akhil Vase | Data & ML Engineer, Baker Hughes
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_COLS = [
    "wellhead_pressure_psi",
    "pump_inlet_temp_f",
    "pump_outlet_temp_f",
    "vibration_rms_mms",
    "flow_rate_bbl_day",
    "motor_current_amps",
    "rotational_speed_rpm",
    # Derived features (computed in feature pipeline)
    "pressure_rate_of_change",
    "vibration_rolling_std_1h",
    "temp_delta",                   # outlet - inlet temp
    "current_to_flow_ratio",
    "vibration_trend_6h",
]

SEQUENCE_LENGTH = 144   # 24 hours at 10-minute intervals
PREDICTION_HORIZON = 288  # 48 hours ahead (label: failure within 48h = 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WellSensorDataset(Dataset):
    """
    Sliding-window dataset over sensor time series.
    Each sample: (sequence_length × n_features) → binary failure label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = SEQUENCE_LENGTH,
        pred_horizon: int = PREDICTION_HORIZON,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        features = df[SENSOR_COLS].values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        elif scaler is not None:
            self.scaler = scaler
            features = scaler.transform(features)
        else:
            self.scaler = None

        self.X = torch.tensor(features, dtype=torch.float32)
        # Label: 1 if equipment failed within prediction_horizon steps
        self.y = torch.tensor(df["failure_within_horizon"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len - self.pred_horizon + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len + self.pred_horizon - 1]
        return x, y


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class EquipmentFailureLSTM(nn.Module):
    """
    Bidirectional LSTM for equipment failure prediction.

    Architecture:
      Bi-LSTM (2 layers) → Dropout → Attention → FC → Sigmoid

    The attention layer weights timesteps by learned importance —
    useful since failure signatures often appear in specific time windows
    (e.g., vibration spike 6–12h before failure).
    """

    def __init__(
        self,
        input_size: int = len(SENSOR_COLS),
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        # Temporal attention
        attn_dim = hidden_size * self.directions
        self.attention = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        returns: (batch,) failure probability
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden*dirs)
        lstm_out = self.dropout(lstm_out)

        # Attention-weighted pooling
        attn_weights = self.attention(lstm_out)         # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*dirs)

        return self.classifier(context).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FailurePredictorTrainer:
    """
    Trains, evaluates, and logs the LSTM failure prediction model.
    """

    def __init__(
        self,
        model: EquipmentFailureLSTM | None = None,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        device: str | None = None,
        mlflow_experiment: str = "predictive-maintenance",
        checkpoint_dir: str = "models/pm",
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = (model or EquipmentFailureLSTM()).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_experiment(mlflow_experiment)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> dict[str, float]:
        train_ds = WellSensorDataset(train_df, fit_scaler=True)
        val_ds = WellSensorDataset(val_df, scaler=train_ds.scaler)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # Weighted loss for class imbalance (~5% failure rate)
        pos_weight = torch.tensor([15.0], device=self.device)
        criterion = nn.BCELoss(reduction="mean")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_auc = 0.0
        best_epoch = 0

        with mlflow.start_run():
            mlflow.log_params({
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "seq_len": SEQUENCE_LENGTH,
                "pred_horizon_h": PREDICTION_HORIZON * 10 // 60,
            })

            for epoch in range(1, self.epochs + 1):
                train_loss = self._train_epoch(train_loader, criterion, optimizer)
                val_auc, val_loss = self._eval_epoch(val_loader, criterion)
                scheduler.step(val_loss)

                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc}, step=epoch)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    self._save_checkpoint("best_model.pt", train_ds.scaler)
                    mlflow.pytorch.log_model(self.model, "lstm_model")

                if epoch % 10 == 0:
                    log.info("Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_auc=%.4f",
                             epoch, self.epochs, train_loss, val_loss, val_auc)

            log.info("Best epoch: %d | Best val AUC: %.4f", best_epoch, best_val_auc)
            mlflow.log_metric("best_val_auc", best_val_auc)

        return {"best_val_auc": best_val_auc, "best_epoch": best_epoch}

    def _train_epoch(self, loader: DataLoader, criterion, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            preds = self.model(X)
            loss = criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _eval_epoch(self, loader: DataLoader, criterion) -> tuple[float, float]:
        self.model.eval()
        all_proba, all_labels, total_loss = [], [], 0.0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                proba = self.model(X)
                total_loss += criterion(proba, y).item()
                all_proba.extend(proba.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        auc = roc_auc_score(all_labels, all_proba) if len(set(all_labels)) > 1 else 0.5
        return auc, total_loss / len(loader)

    def _save_checkpoint(self, filename: str, scaler) -> None:
        checkpoint = {
            "model_state": self.model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def predict(self, sensor_window: np.ndarray) -> float:
        """
        Score a single sensor window (seq_len × n_features).
        Returns failure probability in [0, 1].
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(sensor_window, dtype=torch.float32).unsqueeze(0).to(self.device)
            return float(self.model(x).item())
