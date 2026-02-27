"""
Property prediction heads for structure model backbones.

Attaches a lightweight prediction head (MLP, linear, etc.) on top
of frozen or fine-tuned backbone embeddings to predict scalar
or categorical properties (binding affinity, stability, function, …).

This bridges the gap between:
  - **Structure fine-tuning** (FAPE loss on coordinates)
  - **Classical ML** (sklearn on sequence features)

by using learned representations from structure models as input
features for property prediction.

Usage::

    from molfun.ml.heads import PropertyHead

    head = PropertyHead(
        backbone="openfold",
        head_type="mlp",
        task="regression",
        hidden_dims=[256, 128],
    )
    head.fit(pdb_paths, targets)
    preds = head.predict(new_pdb_paths)
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np


# ------------------------------------------------------------------
# Embedding extraction (backbone-agnostic)
# ------------------------------------------------------------------

def extract_embeddings(
    pdb_paths: list[str],
    backbone: str = "openfold",
    layer: str = "last",
    pooling: str = "mean",
    device: str = "cpu",
    batch_size: int = 4,
) -> np.ndarray:
    """
    Extract per-protein embeddings from a structure model backbone.

    Args:
        pdb_paths: Paths to PDB/CIF files.
        backbone: Model name ("openfold", "protenix", etc.).
        layer: Which layer to extract ("last", "middle", or int).
        pooling: How to pool residue embeddings ("mean", "max", "cls").
        device: "cpu" or "cuda".
        batch_size: Batch size for inference.

    Returns:
        np.ndarray of shape [N, embedding_dim].
    """
    if backbone == "openfold":
        return _extract_openfold(pdb_paths, layer, pooling, device, batch_size)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: 'openfold'"
        )


def _extract_openfold(
    pdb_paths: list[str],
    layer: str,
    pooling: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Extract embeddings from OpenFold."""
    try:
        from molfun.backends.openfold import OpenFoldAdapter
    except ImportError:
        raise ImportError(
            "OpenFold backend not available. Install with: pip install molfun[openfold]"
        )

    adapter = OpenFoldAdapter(device=device)
    adapter.load()

    embeddings = []
    for i in range(0, len(pdb_paths), batch_size):
        batch_paths = pdb_paths[i:i + batch_size]
        for path in batch_paths:
            emb = adapter.embed(path, layer=layer)
            if pooling == "mean":
                pooled = emb.mean(axis=0)
            elif pooling == "max":
                pooled = emb.max(axis=0)
            elif pooling == "cls":
                pooled = emb[0]
            else:
                raise ValueError(f"Unknown pooling: '{pooling}'")
            embeddings.append(pooled)

    return np.vstack(embeddings)


# ------------------------------------------------------------------
# PropertyHead
# ------------------------------------------------------------------

class PropertyHead:
    """
    Property prediction head on top of structure model embeddings.

    Workflow:
        1. Extract embeddings from a backbone (OpenFold, Protenix, …)
        2. Train a head (MLP, linear, sklearn model) on embeddings → property
        3. Predict on new structures

    Args:
        backbone: Structure model name (e.g. "openfold").
        head_type: Type of prediction head:
            - "mlp":     PyTorch MLP (requires torch).
            - "linear":  Ridge regression (sklearn).
            - "rf":      Random Forest (sklearn).
            - "svm":     SVM (sklearn).
        task: "regression" or "classification".
        hidden_dims: Hidden layer sizes for MLP head (default [256, 128]).
        layer: Backbone layer for embeddings ("last", "middle", etc.).
        pooling: Embedding pooling ("mean", "max", "cls").
        device: "cpu" or "cuda".
        **head_params: Extra params passed to the head constructor.
    """

    def __init__(
        self,
        backbone: str = "openfold",
        head_type: str = "mlp",
        task: str = "regression",
        hidden_dims: Optional[list[int]] = None,
        layer: str = "last",
        pooling: str = "mean",
        device: str = "cpu",
        **head_params,
    ):
        self.backbone = backbone
        self.head_type = head_type
        self.task = task
        self.hidden_dims = hidden_dims or [256, 128]
        self.layer = layer
        self.pooling = pooling
        self.device = device
        self.head_params = head_params

        self._head = None
        self._fitted = False

    def fit(
        self,
        pdb_paths: list[str],
        y,
        embeddings: Optional[np.ndarray] = None,
    ):
        """
        Fit the head on structure embeddings.

        Args:
            pdb_paths: Paths to PDB/CIF files (ignored if embeddings given).
            y: Target values.
            embeddings: Pre-computed embeddings [N, D].
                        If None, extracted from backbone.

        Returns:
            self
        """
        if embeddings is None:
            embeddings = extract_embeddings(
                pdb_paths, backbone=self.backbone,
                layer=self.layer, pooling=self.pooling,
                device=self.device,
            )

        y = np.asarray(y)
        self._head = self._build_head(embeddings.shape[1])
        self._fit_head(embeddings, y)
        self._fitted = True
        return self

    def predict(
        self,
        pdb_paths: Optional[list[str]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict properties.

        Args:
            pdb_paths: Paths to PDB/CIF files (ignored if embeddings given).
            embeddings: Pre-computed embeddings [N, D].

        Returns:
            np.ndarray of predictions.
        """
        if not self._fitted:
            raise RuntimeError("Head not fitted yet. Call .fit() first.")

        if embeddings is None:
            if pdb_paths is None:
                raise ValueError("Provide pdb_paths or embeddings")
            embeddings = extract_embeddings(
                pdb_paths, backbone=self.backbone,
                layer=self.layer, pooling=self.pooling,
                device=self.device,
            )

        return self._predict_head(embeddings)

    def _build_head(self, input_dim: int):
        """Construct the head model."""
        if self.head_type == "mlp":
            return self._build_mlp(input_dim)
        elif self.head_type == "linear":
            if self.task == "classification":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=1000, **self.head_params)
            else:
                from sklearn.linear_model import Ridge
                return Ridge(**self.head_params)
        elif self.head_type == "rf":
            if self.task == "classification":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=self.head_params.pop("n_estimators", 500),
                    **self.head_params,
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=self.head_params.pop("n_estimators", 500),
                    **self.head_params,
                )
        elif self.head_type == "svm":
            if self.task == "classification":
                from sklearn.svm import SVC
                return SVC(probability=True, **self.head_params)
            else:
                from sklearn.svm import SVR
                return SVR(**self.head_params)
        else:
            raise ValueError(
                f"Unknown head_type '{self.head_type}'. "
                "Available: 'mlp', 'linear', 'rf', 'svm'"
            )

    def _build_mlp(self, input_dim: int):
        """Build a PyTorch MLP head."""
        import torch
        import torch.nn as nn

        layers = []
        prev = input_dim
        for dim in self.hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(0.1)])
            prev = dim
        out_dim = self.head_params.get("n_classes", 1)
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def _fit_head(self, X: np.ndarray, y: np.ndarray):
        """Train the head."""
        if self.head_type == "mlp":
            self._fit_mlp(X, y)
        else:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            self._head.fit(X_scaled, y)

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray):
        """Train PyTorch MLP with early stopping."""
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        device = torch.device(self.device)
        self._head = self._head.to(device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        if self.task == "classification":
            y_t = torch.tensor(y, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.head_params.get("batch_size", 32), shuffle=True)

        lr = self.head_params.get("lr", 1e-3)
        epochs = self.head_params.get("epochs", 100)
        optimizer = torch.optim.Adam(self._head.parameters(), lr=lr)

        self._head.train()
        for _epoch in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = self._head(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

    def _predict_head(self, X: np.ndarray) -> np.ndarray:
        """Run prediction."""
        if self.head_type == "mlp":
            return self._predict_mlp(X)
        else:
            X_scaled = self._scaler.transform(X)
            return self._head.predict(X_scaled)

    def _predict_mlp(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = torch.device(self.device)
        self._head.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            out = self._head(X_t).cpu().numpy()
        if self.task == "regression":
            return out.squeeze(-1)
        return out.argmax(axis=-1)

    def describe(self) -> dict:
        return {
            "backbone": self.backbone,
            "head_type": self.head_type,
            "task": self.task,
            "hidden_dims": self.hidden_dims,
            "layer": self.layer,
            "pooling": self.pooling,
            "fitted": self._fitted,
        }

    def save(self, path: str) -> None:
        from molfun.ml.io import save_model
        save_model(self, path)

    @classmethod
    def load(cls, path: str):
        from molfun.ml.io import load_model
        return load_model(path)
