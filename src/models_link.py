"""
Link Quality Prediction Models:
- Model A: LinkMLP (Alanazi et al., 2025 - Deterministic 10-neuron ANN)
- Model B: LinkUQNet (Rahul Anand, 2026 - MC Dropout Uncertainty Network)
- Model C: TemporalLinkTransformer (Extension MVP - Sequence Transformer + MC Uncertainty Head)
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for _p in [
    os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
    "C:/Users/Reet/AppData/Local/Programs/Python/Python310/lib/site-packages/torch/lib",
]:
    if os.path.exists(_p):
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass

import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

# ─────────────────────────────────────────────────────────────────────────────
# MODEL A: Base Paper (Alanazi et al., 2025)
# ─────────────────────────────────────────────────────────────────────────────
class LinkMLP(nn.Module):
    """
    Baseline link-quality ANN from Alanazi et al. (Base Paper).
    Architecture: Linear(3, 10) -> ReLU -> Linear(10, 1)
    Output: single logit; sigmoid gives deterministic link success probability.
    No dropout, no uncertainty quantification.
    """
    def __init__(self, in_dim: int = 3, hidden: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_from_metrics(self, rssi: float, snr: float, plr: float, scaler, device: str = 'cpu') -> tuple[float, float]:
        """Returns (mean_prob, 0.0) since Model A has zero uncertainty quantification."""
        x_raw = np.array([[rssi, snr, plr]], dtype=np.float32)
        x_scaled = scaler.transform(x_raw)
        x_t = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        self.eval()
        with torch.no_grad():
            prob = float(torch.sigmoid(self.forward(x_t)).item())
        return prob, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MODEL B: Senior's Work (Rahul Anand, 2026)
# ─────────────────────────────────────────────────────────────────────────────
class LinkUQNet(nn.Module):
    """
    Uncertainty-aware link quality network with MC Dropout (Rahul Anand).
    Architecture: Linear(3, 32) -> ReLU -> Dropout(0.3)
                  -> Linear(32, 32) -> ReLU -> Dropout(0.3) -> Linear(32, 1)
    """
    def __init__(self, in_dim: int = 3, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class UncertaintyModel:
    """
    Wraps LinkUQNet for Monte Carlo Dropout inference (Rahul's Senior Implementation).
    Performs N stochastic forward passes with dropout active at inference time.
    """
    def __init__(self, model: LinkUQNet, scaler, device: str = 'cpu', mc_samples: int = 30):
        self.model = model.to(device)
        self.scaler = scaler
        self.device = device
        self.mc_samples = mc_samples
        self.round_cache: dict[tuple[int, int], tuple[float, float]] = {}

    def clear_cache(self):
        self.round_cache.clear()

    def predict_batch(self, X_scaled: np.ndarray, mc_samples: int | None = None):
        n_mc = int(mc_samples or self.mc_samples)
        x_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        x_batch = x_t.repeat(n_mc, 1)
        self.model.train()  # Keep dropout active
        with torch.no_grad():
            preds = torch.sigmoid(self.model(x_batch)).detach().cpu().numpy().flatten()
        self.model.eval()
        return float(np.mean(preds)), float(np.std(preds))

    def predict_from_metrics(self, rssi: float, snr: float, plr: float, mc_samples: int | None = None,
                             sender_id: int | None = None, receiver_id: int | None = None):
        if sender_id is not None and receiver_id is not None:
            pair_key = (sender_id, receiver_id)
            if pair_key in self.round_cache:
                return self.round_cache[pair_key]
        x_raw = np.array([[rssi, snr, plr]], dtype=np.float32)
        x_scaled = self.scaler.transform(x_raw)
        m, s = self.predict_batch(x_scaled, mc_samples)
        res = (m, s)
        if sender_id is not None and receiver_id is not None:
            self.round_cache[pair_key] = res
        return res


# ─────────────────────────────────────────────────────────────────────────────
# MODEL C: Proposed Extension MVP (Temporal Link Transformer)
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence window of link metrics."""
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TemporalLinkTransformer(nn.Module):
    """
    Sequence-Aware Temporal Transformer for Link Quality Prediction.
    Consumes sliding windows of physical metrics (rssi, snr, plr) over length T=10,
    attends to historical channel trends (e.g. fading, shadow emergence, node drift),
    and outputs predictive probability distribution with MC Dropout uncertainty head.
    """
    def __init__(self, in_dim: int = 3, d_model: int = 32, nhead: int = 2,
                 num_layers: int = 2, dim_feedforward: int = 64,
                 dropout: float = 0.1, mc_dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Risk-aware prediction head with MC Dropout
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(p=mc_dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: shape (batch, seq_len, in_dim)
        Output: shape (batch,) logits
        """
        h = self.input_proj(x_seq)
        h = self.pos_encoder(h)
        h_trans = self.transformer_encoder(h)
        # Global temporal pooling across time dimension
        h_pool = h_trans.mean(dim=1)
        out = self.head(h_pool).squeeze(-1)
        return out


class TemporalUncertaintyModel:
    """
    Wraps TemporalLinkTransformer for sequence-aware MC Dropout inference.
    Maintains a rolling history buffer per link with round-level caching for high-speed simulation.
    """
    def __init__(self, model: nn.Module, scaler, device: str = 'cpu', mc_samples: int = 20, seq_len: int = 10):
        self.model = model.to(device)
        self.scaler = scaler
        self.device = device
        self.mc_samples = mc_samples
        self.seq_len = seq_len
        self.history_buffers: dict[tuple[int, int], list[np.ndarray]] = {}
        self.round_cache: dict[tuple[int, int], tuple[float, float]] = {}

    def clear_cache(self):
        """Clears cached link metrics when node positions change at each round."""
        self.round_cache.clear()

    def update_and_predict(self, sender_id: int, receiver_id: int, rssi: float, snr: float, plr: float) -> tuple[float, float]:
        """
        Updates the historical sequence buffer for a given link pair and returns
        (mean_prob, std_prob) for future link health. Caches evaluation within each round.
        """
        pair_key = (sender_id, receiver_id)
        if pair_key in self.round_cache:
            return self.round_cache[pair_key]

        raw_feat = [rssi, snr, plr]
        scaled_feat = self.scaler.transform([raw_feat])[0]

        if pair_key not in self.history_buffers:
            # Initialize history by replicating initial observation
            self.history_buffers[pair_key] = [scaled_feat.copy() for _ in range(self.seq_len)]
        else:
            buf = self.history_buffers[pair_key]
            buf.pop(0)
            buf.append(scaled_feat)

        seq_array = np.array(self.history_buffers[pair_key], dtype=np.float32)
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        seq_batch = seq_tensor.repeat(self.mc_samples, 1, 1)

        self.model.train()  # Keep dropout active
        with torch.no_grad():
            logits = self.model(seq_batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        self.model.eval()

        mean_p = float(np.mean(probs))
        std_p = float(np.std(probs))
        self.round_cache[pair_key] = (mean_p, std_p)
        return mean_p, std_p


# ─────────────────────────────────────────────────────────────────────────────
# Training Utilities
# ─────────────────────────────────────────────────────────────────────────────
def train_point_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      epochs: int = 60, lr: float = 1e-3, batch_size: int = 128,
                      device: str = 'cpu', patience: int = 10) -> nn.Module:
    """Trains LinkMLP or LinkUQNet with BCEWithLogitsLoss and early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    x_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(x_val_t)
            v_loss = criterion(v_logits, y_val_t).item()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model


def train_temporal_model(model: TemporalLinkTransformer, X_seq_train: np.ndarray, y_seq_train: np.ndarray,
                         X_seq_val: np.ndarray, y_seq_val: np.ndarray,
                         epochs: int = 50, lr: float = 1e-3, batch_size: int = 128,
                         device: str = 'cpu', patience: int = 8) -> TemporalLinkTransformer:
    """Trains TemporalLinkTransformer with BCEWithLogitsLoss."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_seq_train, dtype=torch.float32),
        torch.tensor(y_seq_train, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    x_val_t = torch.tensor(X_seq_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_seq_val, dtype=torch.float32).to(device)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(x_val_t), y_val_t).item()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model
