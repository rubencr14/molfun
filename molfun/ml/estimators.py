"""
Protein-aware sklearn estimator wrappers.

Thin wrappers around sklearn that bundle a ``ProteinFeaturizer`` with
a classifier or regressor into a single object with
``fit(sequences, y)`` / ``predict(sequences)`` / ``score()``.

Internally uses ``sklearn.pipeline.Pipeline`` so the full sklearn
ecosystem (cross-validation, grid search, etc.) works out of the box.

Usage::

    from molfun.ml import ProteinRegressor

    model = ProteinRegressor(estimator="random_forest", n_estimators=500)
    model.fit(sequences, affinities)
    preds = model.predict(new_sequences)
    model.save("model.joblib")
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np

from molfun.ml.features import ProteinFeaturizer, DEFAULT_FEATURES


# ------------------------------------------------------------------
# Estimator factories
# ------------------------------------------------------------------

def _make_classifier(name: str, **kw):
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=kw.pop("n_estimators", 500), **kw)
    if name == "svm":
        from sklearn.svm import SVC
        return SVC(kernel=kw.pop("kernel", "rbf"), probability=True, **kw)
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=kw.pop("max_iter", 1000), **kw)
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**kw)
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**kw)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", **kw)
    raise ValueError(f"Unknown classifier: '{name}'. Available: {CLASSIFIER_NAMES}")


def _make_regressor(name: str, **kw):
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=kw.pop("n_estimators", 500), **kw)
    if name == "svm":
        from sklearn.svm import SVR
        return SVR(kernel=kw.pop("kernel", "rbf"), **kw)
    if name == "linear":
        from sklearn.linear_model import Ridge
        return Ridge(**kw)
    if name == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(max_iter=kw.pop("max_iter", 5000), **kw)
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**kw)
    if name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(**kw)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**kw)
    raise ValueError(f"Unknown regressor: '{name}'. Available: {REGRESSOR_NAMES}")


CLASSIFIER_NAMES = ["random_forest", "svm", "logistic", "gradient_boosting", "knn", "xgboost"]
REGRESSOR_NAMES = ["random_forest", "svm", "linear", "lasso", "gradient_boosting", "knn", "xgboost"]


# ------------------------------------------------------------------
# Base wrapper
# ------------------------------------------------------------------

class _ProteinEstimatorBase:
    """Shared logic for classifier and regressor wrappers."""

    _make_fn = None  # override in subclass

    def __init__(
        self,
        estimator: str = "random_forest",
        featurizer: Optional[ProteinFeaturizer] = None,
        features: Optional[list[str]] = None,
        scale: bool = True,
        **estimator_params,
    ):
        """
        Args:
            estimator: Name of the sklearn estimator.
            featurizer: Pre-configured ``ProteinFeaturizer`` instance.
                        Mutually exclusive with ``features``.
            features: List of feature names (shorthand for creating a featurizer).
            scale: If True, apply StandardScaler before the estimator.
            **estimator_params: Passed directly to the sklearn estimator constructor.
        """
        if featurizer is not None and features is not None:
            raise ValueError("Provide either 'featurizer' or 'features', not both.")
        self.estimator_name = estimator
        self.featurizer = featurizer or ProteinFeaturizer(features=features)
        self.scale = scale
        self.estimator_params = estimator_params
        self._pipeline = None

    def fit(self, X, y):
        """
        Fit the model.

        Args:
            X: List of protein sequences (str) or numpy feature matrix.
            y: Target array (labels for classification, values for regression).

        Returns:
            self
        """
        from sklearn.pipeline import Pipeline as SkPipeline

        steps = [("featurize", self.featurizer)]
        if self.scale:
            from sklearn.preprocessing import StandardScaler
            steps.append(("scale", StandardScaler()))
        steps.append(("model", self._make_fn(self.estimator_name, **self.estimator_params)))

        self._pipeline = SkPipeline(steps)
        self._pipeline.fit(X, y)
        return self

    def _fit_precomputed(self, X_features, y):
        """
        Fit from a pre-computed feature matrix (skip the featurizer step).

        Used by pipeline steps when featurize_step already ran.
        """
        from sklearn.pipeline import Pipeline as SkPipeline

        steps = []
        if self.scale:
            from sklearn.preprocessing import StandardScaler
            steps.append(("scale", StandardScaler()))
        steps.append(("model", self._make_fn(self.estimator_name, **self.estimator_params)))

        self._pipeline = SkPipeline(steps) if steps else None
        self._pipeline.fit(X_features, y)
        return self

    def predict(self, X):
        """Predict targets for new sequences."""
        self._check_fitted()
        return self._pipeline.predict(X)

    def score(self, X, y) -> float:
        """Score on test data (accuracy for classifiers, R² for regressors)."""
        self._check_fitted()
        return self._pipeline.score(X, y)

    def transform(self, X) -> np.ndarray:
        """Extract features without predicting (featurizer + scaler only)."""
        self._check_fitted()
        pipe = self._pipeline
        result = pipe.named_steps["featurize"].transform(X)
        if "scale" in pipe.named_steps:
            result = pipe.named_steps["scale"].transform(result)
        return result

    @property
    def sklearn_pipeline(self):
        """Access the underlying sklearn Pipeline for advanced use."""
        return self._pipeline

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Feature importances if the estimator supports them."""
        if self._pipeline is None:
            return None
        model = self._pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        if hasattr(model, "coef_"):
            return np.abs(model.coef_).flatten()
        return None

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return the top-n most important features by name."""
        imp = self.feature_importances
        if imp is None:
            return []
        names = self.featurizer.feature_names
        if len(names) != len(imp):
            names = [f"feature_{i}" for i in range(len(imp))]
        pairs = sorted(zip(names, imp), key=lambda x: -x[1])
        return pairs[:n]

    def describe(self) -> dict:
        """Serializable description of the model config."""
        return {
            "estimator": self.estimator_name,
            "features": self.featurizer.features,
            "n_features": self.featurizer.n_features,
            "scale": self.scale,
            "estimator_params": self.estimator_params,
            "fitted": self._pipeline is not None,
        }

    def save(self, path: str) -> None:
        """Save the full model (featurizer + scaler + estimator) with joblib."""
        from molfun.ml.io import save_model
        save_model(self, path)

    @classmethod
    def load(cls, path: str):
        """Load a saved model."""
        from molfun.ml.io import load_model
        return load_model(path)

    def _check_fitted(self):
        if self._pipeline is None:
            raise RuntimeError("Model not fitted yet. Call .fit(X, y) first.")

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator_name,
            "features": self.featurizer.features,
            "scale": self.scale,
            **self.estimator_params,
        }


# ------------------------------------------------------------------
# Public classes
# ------------------------------------------------------------------

class ProteinClassifier(_ProteinEstimatorBase):
    """
    Sklearn-compatible protein classifier.

    Wraps a featurizer + optional scaler + sklearn classifier into a
    single object.

    Available estimators: random_forest, svm, logistic, gradient_boosting,
    knn, xgboost.

    Usage::

        clf = ProteinClassifier(estimator="random_forest", n_estimators=500)
        clf.fit(sequences, labels)
        preds = clf.predict(new_sequences)
        proba = clf.predict_proba(new_sequences)
    """
    _make_fn = staticmethod(_make_classifier)

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities (if estimator supports them)."""
        self._check_fitted()
        return self._pipeline.predict_proba(X)


class ProteinRegressor(_ProteinEstimatorBase):
    """
    Sklearn-compatible protein regressor.

    Wraps a featurizer + optional scaler + sklearn regressor into a
    single object.

    Available estimators: random_forest, svm, linear, lasso,
    gradient_boosting, knn, xgboost.

    Usage::

        reg = ProteinRegressor(estimator="random_forest")
        reg.fit(sequences, affinities)
        preds = reg.predict(new_sequences)
    """
    _make_fn = staticmethod(_make_regressor)

    def evaluate(self, X, y) -> dict[str, float]:
        """
        Compute regression metrics: MAE, RMSE, R², Pearson.

        Args:
            X: Test sequences.
            y: True target values.

        Returns:
            Dict with metric names and values.
        """
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        preds = np.asarray(preds, dtype=np.float64)

        mae = np.abs(preds - y).mean()
        rmse = np.sqrt(((preds - y) ** 2).mean())
        r2 = self.score(X, y)

        pearson = 0.0
        if len(y) > 2:
            vp = preds - preds.mean()
            vt = y - y.mean()
            denom = np.linalg.norm(vp) * np.linalg.norm(vt)
            if denom > 1e-8:
                pearson = float(np.dot(vp, vt) / denom)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "pearson": pearson,
            "n_samples": len(y),
        }
