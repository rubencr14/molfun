"""
Serialization for ML models (joblib-based).
"""

from __future__ import annotations
from pathlib import Path


def save_model(model, path: str) -> None:
    """Save a fitted ProteinClassifier/ProteinRegressor to disk."""
    import joblib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Load a saved ProteinClassifier/ProteinRegressor from disk."""
    import joblib
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
