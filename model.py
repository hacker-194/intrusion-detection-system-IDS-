import collections
import copy
import functools
import logging
import math
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from river.drift import ADWIN
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger("ids_core")

# Mapping from CICFlowMeter column names to NFStream 6.x attribute names.
CIC_TO_NFSTREAM: Dict[str, str] = {
    "Flow Duration": "bidirectional_duration_ms",
    "Total Fwd Packets": "src2dst_packets",
    "Total Length of Fwd Packets": "src2dst_bytes",
    "Fwd Packet Length Max": "src2dst_max_ps",
    "Fwd Packet Length Min": "src2dst_min_ps",
    "Fwd Packet Length Mean": "src2dst_mean_ps",
    "Fwd Packet Length Std": "src2dst_stddev_ps",
    "Bwd Packet Length Max": "dst2src_max_ps",
    "Bwd Packet Length Min": "dst2src_min_ps",
    "Bwd Packet Length Mean": "dst2src_mean_ps",
    "Bwd Packet Length Std": "dst2src_stddev_ps",
    "Flow Bytes/s": "bidirectional_bytes_per_sec",
    "Flow Packets/s": "bidirectional_packets_per_sec",
    "Flow IAT Mean": "bidirectional_mean_piat_ms",
    "Flow IAT Std": "bidirectional_stddev_piat_ms",
    "Flow IAT Max": "bidirectional_max_piat_ms",
    "Flow IAT Min": "bidirectional_min_piat_ms",
    "Fwd IAT Total": "src2dst_duration_ms",
    "Fwd IAT Mean": "src2dst_mean_piat_ms",
    "Fwd IAT Std": "src2dst_stddev_piat_ms",
    "Fwd IAT Max": "src2dst_max_piat_ms",
    "Fwd IAT Min": "src2dst_min_piat_ms",
    "Bwd IAT Total": "dst2src_duration_ms",
    "Bwd IAT Mean": "dst2src_mean_piat_ms",
    "Bwd IAT Std": "dst2src_stddev_piat_ms",
    "Bwd IAT Max": "dst2src_max_piat_ms",
    "Bwd IAT Min": "dst2src_min_piat_ms",
    "Fwd Header Length": "src2dst_header_bytes",
    "Bwd Header Length": "dst2src_header_bytes",
    "Fwd Packets/s": "src2dst_packets_per_sec",
    "Bwd Packets/s": "dst2src_packets_per_sec",
    "Min Packet Length": "bidirectional_min_ps",
    "Max Packet Length": "bidirectional_max_ps",
    "Packet Length Mean": "bidirectional_mean_ps",
    "Packet Length Std": "bidirectional_stddev_ps",
    "Packet Length Variance": "bidirectional_variance_ps",
    "FIN Flag Count": "bidirectional_fin_packets",
    "PSH Flag Count": "bidirectional_psh_packets",
    "ACK Flag Count": "bidirectional_ack_packets",
    "Average Packet Size": "bidirectional_avg_ps",
    "Subflow Fwd Bytes": "src2dst_subflow_bytes",
    "Init_Win_bytes_forward": "src2dst_tcp_init_window",
    "Init_Win_bytes_backward": "dst2src_tcp_init_window",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "min_seg_size_forward": "src2dst_min_seg_size",
    "Active Mean": "bidirectional_active_mean",
    "Active Max": "bidirectional_active_max",
    "Active Min": "bidirectional_active_min",
    "Idle Mean": "bidirectional_idle_mean",
    "Idle Max": "bidirectional_idle_max",
    "Idle Min": "bidirectional_idle_min",
}

_NFSTREAM_COMPUTED: Dict[str, Any] = {
    "bidirectional_bytes_per_sec": lambda f: f.bidirectional_bytes / max(f.bidirectional_duration_ms / 1000.0, 0.001),
    "bidirectional_packets_per_sec": lambda f: f.bidirectional_packets / max(f.bidirectional_duration_ms / 1000.0, 0.001),
    "src2dst_packets_per_sec": lambda f: f.src2dst_packets / max(f.bidirectional_duration_ms / 1000.0, 0.001),
    "dst2src_packets_per_sec": lambda f: f.dst2src_packets / max(f.bidirectional_duration_ms / 1000.0, 0.001),
    "src2dst_header_bytes": lambda f: getattr(f, "src2dst_header_bytes", 0.0),
    "dst2src_header_bytes": lambda f: getattr(f, "dst2src_header_bytes", 0.0),
    "bidirectional_variance_ps": lambda f: (getattr(f, "bidirectional_stddev_ps", 0.0) or 0.0) ** 2,
    "bidirectional_avg_ps": lambda f: f.bidirectional_bytes / max(f.bidirectional_packets, 1),
    "src2dst_tcp_init_window": lambda f: getattr(f, "src2dst_tcp_init_window", 0.0),
    "dst2src_tcp_init_window": lambda f: getattr(f, "dst2src_tcp_init_window", 0.0),
    "act_data_pkt_fwd": lambda f: getattr(getattr(f, "udps", None), "act_data_pkt_fwd", 0.0),
    "src2dst_min_seg_size": lambda f: getattr(f, "src2dst_min_seg_size", 0.0),
    "src2dst_subflow_bytes": lambda f: f.src2dst_bytes,
    "bidirectional_active_mean": lambda f: getattr(f, "bidirectional_active_mean", 0.0),
    "bidirectional_active_max": lambda f: getattr(f, "bidirectional_active_max", 0.0),
    "bidirectional_active_min": lambda f: getattr(f, "bidirectional_active_min", 0.0),
    "bidirectional_idle_mean": lambda f: getattr(f, "bidirectional_idle_mean", 0.0),
    "bidirectional_idle_max": lambda f: getattr(f, "bidirectional_idle_max", 0.0),
    "bidirectional_idle_min": lambda f: getattr(f, "bidirectional_idle_min", 0.0),
}


def _stable_sigmoid(x):
    """Numerically stable sigmoid, clipping to [-500, 500] to avoid overflow."""
    x = np.clip(np.asarray(x, dtype=np.float64), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))
  
_AE_HIGH_CONF = 0.7
_AE_LOW_CONF = 0.3


# Autoencoder (PyTorch)
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



# RealTimeIDS (autoencoder wrapper with scaler + threshold)
class RealTimeIDS:
    def __init__(self, input_dim: int, feature_names: List[str]):
        self.input_dim = input_dim
        self.feature_names = feature_names
        self._model: Optional[Autoencoder] = None
        self._scaler: Optional[StandardScaler] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self._threshold: Optional[float] = None
        self._device = torch.device("cpu")
        self._lock = threading.RLock()
        self._retrain_lock = threading.Lock()
        self._feature_version: Optional[str] = None
        self._error_count = 0
        self._latent_dim: Optional[int] = None

    def init_model(self, latent_dim: int = 16):
        with self._lock:
            self._latent_dim = latent_dim
            self._model = Autoencoder(self.input_dim, latent_dim).to(self._device)
            self._scaler = StandardScaler()
            self._optimizer = optim.Adam(self._model.parameters(), lr=1e-3)
            self._threshold = 0.5

    def get_status(self):
        with self._lock:
            return {
                "model_ready": self._model is not None,
                "threshold": self._threshold,
                "feature_version": self._feature_version or "",
                "error_count": self._error_count,
                "scaler_fitted": hasattr(self._scaler, "mean_") if self._scaler else False,
            }

    def extract_features(self, flow) -> np.ndarray:
        values = []
        for name in self.feature_names:
            val = 0.0
            try:
                nf_computed = _NFSTREAM_COMPUTED.get(name)
                if nf_computed is not None:
                    val = float(nf_computed(flow))
                else:
                    val = float(getattr(flow, name, 0.0))
            except (TypeError, ValueError, AttributeError):
                val = 0.0
            if not math.isfinite(val):
                val = 0.0
            values.append(val)
        return np.array(values, dtype=np.float32)

    def predict_with_confidence(self, features: np.ndarray) -> Tuple[int, float, float]:
        with self._lock:
            if self._model is None or self._scaler is None:
                raise RuntimeError("Autoencoder not initialized")
            if not hasattr(self._scaler, "mean_"):
                raise RuntimeError("Scaler not fitted -- call retrain() first.")
            try:
                self._model.eval()
                x = features.reshape(1, -1)
                x_scaled = self._scaler.transform(x)
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self._device)
                with torch.no_grad():
                    recon = self._model(x_tensor)
                error = float(torch.mean((x_tensor - recon) ** 2).cpu().numpy())
                scale = max(self._threshold, 1e-6)
                confidence = float(_stable_sigmoid((error - self._threshold) / scale))
                prediction = 1 if error > self._threshold else 0
                return prediction, confidence, error
            except Exception:
                self._error_count += 1
                raise

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch predict: returns (predictions, confidences, errors) as 1-D arrays."""
        with self._lock:
            if self._model is None or self._scaler is None:
                raise RuntimeError("Autoencoder not initialized")
            if not hasattr(self._scaler, "mean_"):
                raise RuntimeError("Scaler not fitted -- call retrain() first.")
            try:
                self._model.eval()
                Xs = self._scaler.transform(X)
                tensor = torch.tensor(Xs, dtype=torch.float32).to(self._device)
                with torch.no_grad():
                    recon = self._model(tensor)
                errors = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()
                scale = max(self._threshold, 1e-6)
                confidences = _stable_sigmoid((errors - self._threshold) / scale)
                predictions = (errors > self._threshold).astype(np.int32)
                return predictions, confidences.astype(np.float64), errors.astype(np.float64)
            except Exception:
                self._error_count += 1
                raise

    def clone(self):
        with self._lock:
            new = RealTimeIDS(self.input_dim, list(self.feature_names))
            if self._model:
                new._model = copy.deepcopy(self._model).to(self._device)
                new._latent_dim = self._latent_dim
            if self._scaler:
                new._scaler = copy.deepcopy(self._scaler)
            new._threshold = self._threshold
            new._feature_version = self._feature_version
            new._optimizer = optim.Adam(new._model.parameters(), lr=1e-3) if new._model else None
            return new

    def save(self, path: str):
        with self._lock:
            latent_dim = self._latent_dim if self._latent_dim is not None else 16
            state = {
                "model": self._model.state_dict() if self._model else None,
                "scaler": self._scaler,
                "threshold": self._threshold,
                "feature_names": self.feature_names,
                "input_dim": self.input_dim,
                "feature_version": self._feature_version,
                "latent_dim": latent_dim,
            }
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(suffix=".joblib", dir=Path(path).parent)
            os.close(fd)
            try:
                joblib.dump(state, tmp_path)
                os.replace(tmp_path, path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

    def load(self, path: str):
        _orig_torch_load = torch.load
        torch.load = functools.partial(_orig_torch_load, map_location="cpu", weights_only=False)
        try:
            state = joblib.load(path)
        finally:
            torch.load = _orig_torch_load
        self.feature_names = state["feature_names"]
        self.input_dim = state["input_dim"]
        self._scaler = state["scaler"]
        self._threshold = state["threshold"]
        self._feature_version = state.get("feature_version")
        latent_dim = state.get("latent_dim", 16)
        self._latent_dim = latent_dim
        self._model = Autoencoder(self.input_dim, latent_dim).to(self._device)
        self._model.load_state_dict(state["model"])
        self._optimizer = optim.Adam(self._model.parameters(), lr=1e-3)

    def retrain(self, X: np.ndarray, y: np.ndarray,
                epochs: int = 5, batch_size: int = 64,
                feature_version: Optional[str] = None, val_frac: float = 0.15):
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got X={len(X)}, y={len(y)}")
        with self._retrain_lock:
            with self._lock:
                if self._model is None:
                    raise RuntimeError("Model not initialized")
                latent_dim = (
                    self._latent_dim
                    if self._latent_dim is not None
                    else self._model.encoder[2].out_features
                )
            benign = X[y == 0]
            if len(benign) < 2:
                raise ValueError(
                    f"Autoencoder retrain needs at least 2 benign samples, got {len(benign)}."
                )
            min_val_frac = 1.0 / len(benign)
            max_val_frac = 1.0 - (1.0 / len(benign))
            effective_val_frac = min(max(val_frac, min_val_frac), max_val_frac)
            X_b_train, X_b_val = train_test_split(benign, test_size=effective_val_frac, random_state=42)
            new_scaler = StandardScaler().fit(X_b_train)
            Xs_train = new_scaler.transform(X_b_train)
            tensor = torch.tensor(Xs_train, dtype=torch.float32).to(self._device)
            new_model = Autoencoder(self.input_dim, latent_dim).to(self._device)
            new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            new_model.train()
            for _ in range(epochs):
                for i in range(0, len(tensor), batch_size):
                    batch = tensor[i:i + batch_size]
                    new_optimizer.zero_grad()
                    recon = new_model(batch)
                    loss = criterion(recon, batch)
                    loss.backward()
                    new_optimizer.step()
            new_model.eval()
            Xs_val = new_scaler.transform(X_b_val)
            val_tensor = torch.tensor(Xs_val, dtype=torch.float32).to(self._device)
            with torch.no_grad():
                val_errors = torch.mean((val_tensor - new_model(val_tensor)) ** 2, dim=1).cpu().numpy()
            new_threshold = float(np.mean(val_errors) + 2 * np.std(val_errors))
            with self._lock:
                self._model = new_model
                self._optimizer = new_optimizer
                self._scaler = new_scaler
                self._threshold = new_threshold
                self._latent_dim = latent_dim
                if feature_version:
                    self._feature_version = feature_version
        return self


# LightGBM classifier
class LightGBMClassifier:
    """Binary classifier built on a LightGBM booster."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.input_dim = len(feature_names)
        self._model: Optional[lgb.Booster] = None
        self._lock = threading.RLock()
        self._error_count = 0

    def load(self, path: str):
        with self._lock:
            self._model = lgb.Booster(model_file=path)

    def save(self, path: str):
        with self._lock:
            if self._model:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(suffix=".txt", dir=Path(path).parent)
                os.close(fd)
                try:
                    self._model.save_model(tmp_path)
                    os.replace(tmp_path, path)
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[int, float, float]:
        with self._lock:
            if self._model is None:
                raise RuntimeError("LightGBM not loaded")
            try:
                raw = self._model.predict(X.reshape(1, -1), raw_score=True)
                raw_val = float(raw[0])
                conf = float(_stable_sigmoid(raw_val))
                pred = 1 if conf >= 0.5 else 0
                return pred, conf, raw_val
            except Exception:
                self._error_count += 1
                raise

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch predict: returns (predictions, confidences, raw_scores) as 1-D arrays."""
        with self._lock:
            if self._model is None:
                raise RuntimeError("LightGBM not loaded")
            try:
                raw_scores = self._model.predict(X, raw_score=True)
                confidences = _stable_sigmoid(raw_scores)
                predictions = (confidences >= 0.5).astype(np.int32)
                return predictions, confidences.astype(np.float64), raw_scores.astype(np.float64)
            except Exception:
                self._error_count += 1
                raise

    def clone(self):
        with self._lock:
            new = LightGBMClassifier(self.feature_names)
            if self._model:
                tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
                tmp_name = tmp.name
                tmp.close()
                try:
                    self._model.save_model(tmp_name)
                    new._model = lgb.Booster(model_file=tmp_name)
                finally:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
            return new

    def retrain(self, X: np.ndarray, y: np.ndarray):
        if len(np.unique(y)) < 2:
            raise ValueError("Need both classes for LightGBM")
        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "max_depth": -1,
        }
        min_class_count = int(np.bincount(y.astype(int)).min())
        if min_class_count < 5:
            logger.warning(
                "LightGBM retrain: minority class has only %d samples -- "
                "skipping internal validation split.",
                min_class_count,
            )
            train_data = lgb.Dataset(X, label=y)
            new_model = lgb.train(params, train_data, num_boost_round=100)
        else:
            Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            train_data = lgb.Dataset(Xtr, label=ytr)
            val_data = lgb.Dataset(Xval, label=yval)
            new_model = lgb.train(
                params, train_data, num_boost_round=300,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(10)],
            )
        with self._lock:
            self._model = new_model



# HybridIDS 
class HybridIDS:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.autoencoder: Optional[RealTimeIDS] = None
        self.lightgbm: Optional[LightGBMClassifier] = None
        self.meta_model: Optional[lgb.Booster] = None
        self._calib_a: Optional[float] = None
        self._calib_b: Optional[float] = None
        self._lock = threading.RLock()
        self._lgb_explainer: Optional[Any] = None
        self._meta_explainer: Optional[Any] = None
        self._shap_background_data: Optional[np.ndarray] = None

    @property
    def input_dim(self) -> int:
        return self.autoencoder.input_dim if self.autoencoder else 0

    def load(self, ae_path: str, lgb_path: str,
             meta_path: Optional[str] = None, calib_path: Optional[str] = None):
        self.autoencoder = RealTimeIDS(len(self.feature_names), self.feature_names)
        self.autoencoder.load(ae_path)
        self.feature_names = self.autoencoder.feature_names
        self.lightgbm = LightGBMClassifier(self.feature_names)
        self.lightgbm.load(lgb_path)
        if meta_path and Path(meta_path).exists():
            self.meta_model = lgb.Booster(model_file=meta_path)
        else:
            self.meta_model = None
        if calib_path and Path(calib_path).exists():
            calib_data = joblib.load(calib_path)
            self._calib_a = calib_data.get("a")
            self._calib_b = calib_data.get("b")
        else:
            self._calib_a = None
            self._calib_b = None
        self._init_shap_explainers()

    def _init_shap_explainers(self):
        if not SHAP_AVAILABLE:
            return
        if self.lightgbm is None or self.lightgbm._model is None:
            return
        try:
            bg = self._shap_background_data or np.zeros((1, self.lightgbm.input_dim))
            self._lgb_explainer = shap.TreeExplainer(
                self.lightgbm._model, data=bg, model_output="raw",
            )
            if self.meta_model is not None:
                self._meta_explainer = shap.TreeExplainer(
                    self.meta_model, data=np.zeros((1, 5)), model_output="raw",
                )
        except Exception as e:
            logger.warning("Failed to create SHAP explainers: %s", e)
            self._lgb_explainer = None
            self._meta_explainer = None

    def save(self, ae_path: str, lgb_path: str, meta_path: str, calib_path: str):
        if self.autoencoder:
            self.autoencoder.save(ae_path)
        if self.lightgbm:
            self.lightgbm.save(lgb_path)
        if self.meta_model is not None:
            meta_dir = Path(meta_path).parent
            meta_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(suffix=".txt", dir=meta_dir)
            os.close(fd)
            try:
                self.meta_model.save_model(tmp_path)
                os.replace(tmp_path, meta_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        if self._calib_a is not None and self._calib_b is not None:
            calib_dir = Path(calib_path).parent
            calib_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_calib = tempfile.mkstemp(suffix=".joblib", dir=calib_dir)
            os.close(fd)
            try:
                joblib.dump({"a": self._calib_a, "b": self._calib_b}, tmp_calib)
                os.replace(tmp_calib, calib_path)
            except Exception:
                try:
                    os.unlink(tmp_calib)
                except OSError:
                    pass
                raise

    def extract_features(self, flow):
        return self.autoencoder.extract_features(flow)

    # -- public inference API 

    def predict_with_confidence(self, features: np.ndarray) -> Tuple[int, float]:
        pred, conf, _ = self._predict_with_confidence_impl(features)
        return pred, conf

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds, confs, _ = self._predict_batch_impl(X)
        return preds, confs

    def predict_batch_with_drift(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._predict_batch_impl(X)

    def get_prediction_with_drift(self, features: np.ndarray) -> Tuple[int, float, float]:
        return self._predict_with_confidence_impl(features)

    def get_autoencoder_error(self, features: np.ndarray) -> float:
        with self._lock:
            ae = self.autoencoder
        if ae is None:
            raise RuntimeError("Autoencoder not loaded")
        _, _, error = ae.predict_with_confidence(features)
        return error

    # -- internals 

    def _predict_with_confidence_impl(self, features: np.ndarray) -> Tuple[int, float, float]:
        with self._lock:
            ae = self.autoencoder
            lgb_clf = self.lightgbm
            meta = self.meta_model
            calib_a = self._calib_a
            calib_b = self._calib_b
            if ae is None or lgb_clf is None:
                raise RuntimeError("Hybrid model not fully loaded")

            ae_pred, ae_conf, ae_error = ae.predict_with_confidence(features)
            if ae_conf >= _AE_HIGH_CONF:
                final_pred = ae_pred
                final_conf = ae_conf
            elif ae_conf <= _AE_LOW_CONF:
                final_pred = ae_pred
                final_conf = ae_conf
            else:
                lgb_pred, lgb_conf, lgb_raw = lgb_clf.predict_with_confidence(features)
                meta_input = np.array(
                    [[ae_conf, ae_error, lgb_conf, ae_pred, lgb_pred]],
                    dtype=np.float32,
                )
                if meta is not None:
                    raw_meta = meta.predict(meta_input, raw_score=True)[0]
                    if calib_a is not None and calib_b is not None:
                        final_conf = float(_stable_sigmoid(calib_a * raw_meta + calib_b))
                    else:
                        final_conf = float(_stable_sigmoid(raw_meta))
                else:
                    final_conf = 0.5 * ae_conf + 0.5 * lgb_conf
                final_pred = 1 if final_conf >= 0.5 else 0

            return final_pred, float(final_conf), float(ae_error)

    def _predict_batch_impl(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            ae = self.autoencoder
            lgb_clf = self.lightgbm
            meta = self.meta_model
            calib_a = self._calib_a
            calib_b = self._calib_b
            if ae is None or lgb_clf is None:
                raise RuntimeError("Hybrid model not fully loaded")

            ae_preds, ae_confs, ae_errors = ae.predict_batch(X)
            lgb_preds, lgb_confs, _ = lgb_clf.predict_batch(X)

            if meta is not None:
                meta_X = np.column_stack(
                    [ae_confs, ae_errors, lgb_confs, ae_preds.astype(np.float32), lgb_preds.astype(np.float32)]
                ).astype(np.float32)
                raw_meta = meta.predict(meta_X, raw_score=True)
                if calib_a is not None and calib_b is not None:
                    meta_confs = _stable_sigmoid(calib_a * raw_meta + calib_b)
                else:
                    meta_confs = _stable_sigmoid(raw_meta)
            else:
                meta_confs = 0.5 * ae_confs + 0.5 * lgb_confs

            meta_preds = (meta_confs >= 0.5).astype(np.int32)
            ae_is_certain_attack = ae_confs >= _AE_HIGH_CONF
            ae_is_certain_benign = ae_confs <= _AE_LOW_CONF
            ae_is_certain = ae_is_certain_attack | ae_is_certain_benign

            final_preds = np.where(ae_is_certain, ae_preds, meta_preds)
            final_confs = np.where(ae_is_certain, ae_confs, meta_confs)

            return final_preds, final_confs, ae_errors

    # -- SHAP explainability 

    def explain(self, features: np.ndarray) -> Optional[Dict[str, Any]]:
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed -- cannot provide explanations.")
            return None
        if features.ndim == 1:
            features = features.reshape(1, -1)
        with self._lock:
            ae = self.autoencoder
            lgb_clf = self.lightgbm
            meta = self.meta_model
            lgb_explainer = self._lgb_explainer
            meta_explainer = self._meta_explainer
            if ae is None or lgb_clf is None:
                raise RuntimeError("Model not fully loaded")
            if lgb_explainer is None:
                return None
            ae_pred, ae_conf, ae_error = ae.predict_with_confidence(features[0])
            lgb_pred, lgb_conf, lgb_raw = lgb_clf.predict_with_confidence(features[0])
        try:
            lgb_shap_values = lgb_explainer.shap_values(features, check_additivity=False)
            lgb_base_value = lgb_explainer.expected_value
        except Exception as e:
            logger.warning("Failed to compute SHAP for LightGBM: %s", e)
            lgb_shap_values = None
            lgb_base_value = None
        meta_shap_values = None
        meta_base_value = None
        if meta is not None and meta_explainer is not None:
            meta_input = np.array([[ae_conf, ae_error, lgb_conf, ae_pred, lgb_pred]], dtype=np.float32)
            try:
                meta_shap_values = meta_explainer.shap_values(meta_input, check_additivity=False)
                meta_base_value = meta_explainer.expected_value
            except Exception as e:
                logger.warning("Failed to compute SHAP for meta-model: %s", e)
        result: Dict[str, Any] = {
            "feature_names": self.feature_names,
            "lgb_shap_values": lgb_shap_values[0].tolist() if lgb_shap_values is not None else None,
            "lgb_base_value": float(lgb_base_value) if lgb_base_value is not None else None,
        }
        if meta_shap_values is not None:
            result["meta_feature_names"] = ["ae_conf", "ae_error", "lgb_conf", "ae_pred", "lgb_pred"]
            result["meta_shap_values"] = meta_shap_values[0].tolist()
            result["meta_base_value"] = float(meta_base_value) if meta_base_value is not None else None
        return result



    def retrain(self, X: np.ndarray, y: np.ndarray,
                epochs: int = 5, batch_size: int = 64,
                feature_version: Optional[str] = None, n_folds: int = 5) -> "HybridIDS":
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got X={len(X)}, y={len(y)}")
        with self._lock:
            if self.autoencoder is None or self.lightgbm is None:
                raise RuntimeError("HybridIDS cannot retrain: base models not loaded.")
        TEST_SIZE = 0.2
        min_required_per_class = math.ceil(1.0 / min(TEST_SIZE, 1.0 - TEST_SIZE))
        class_counts = np.bincount(y.astype(int))
        if len(class_counts) < 2 or class_counts.min() < min_required_per_class:
            counts_str = ", ".join(f"class {c}: {n}" for c, n in enumerate(class_counts))
            raise ValueError(
                f"Retrain buffer needs at least {min_required_per_class} samples "
                f"of each class. Got: {counts_str}."
            )
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, stratify=y,
        )
        bg_size = min(100, np.sum(y_train == 0))
        self._shap_background_data = X_train[y_train == 0][:bg_size] if bg_size > 0 else None
        with self._lock:
            new_ae = self.autoencoder.clone()
            new_lgb = self.lightgbm.clone()
        new_ae.retrain(X_train, y_train, epochs=epochs, batch_size=batch_size, feature_version=feature_version)
        new_lgb.retrain(X_train, y_train)

        MIN_FOLD_TRAIN_MINORITY = 5
        min_class_count = int(np.bincount(y_train.astype(int)).min())
        use_stacking = True
        safe_n_folds = n_folds
        for k in range(min(n_folds, min_class_count), 1, -1):
            if (min_class_count * (k - 1)) // k >= MIN_FOLD_TRAIN_MINORITY:
                safe_n_folds = k
                break
        else:
            use_stacking = False

        if not use_stacking:
            logger.warning("Minority class has only %d samples -- skipping stacking.", min_class_count)
            with self._lock:
                self.autoencoder = new_ae
                self.lightgbm = new_lgb
                self.meta_model = None
                self._calib_a = None
                self._calib_b = None
            threading.Thread(target=self._init_shap_explainers, daemon=True).start()
            return self

        if safe_n_folds < n_folds:
            logger.warning("Clamping OOF folds from %d to %d.", n_folds, safe_n_folds)

        skf = StratifiedKFold(n_splits=safe_n_folds, shuffle=True, random_state=42)
        ae_confs = np.zeros(len(X_train))
        ae_errors = np.zeros(len(X_train))
        lgb_confs = np.zeros(len(X_train))
        ae_preds = np.zeros(len(X_train))
        lgb_preds = np.zeros(len(X_train))
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr_fold = X_train[train_idx]
            y_tr_fold = y_train[train_idx]
            fold_ae = new_ae.clone()
            fold_lgb = new_lgb.clone()
            fold_ae.retrain(X_tr_fold, y_tr_fold, epochs=epochs, batch_size=batch_size, feature_version=feature_version)
            fold_lgb.retrain(X_tr_fold, y_tr_fold)
            for _i, idx in enumerate(val_idx):
                ae_p, ae_c, ae_e = fold_ae.predict_with_confidence(X_train[idx])
                lgb_p, lgb_c, _ = fold_lgb.predict_with_confidence(X_train[idx])
                ae_confs[idx] = ae_c
                ae_errors[idx] = ae_e
                ae_preds[idx] = ae_p
                lgb_confs[idx] = lgb_c
                lgb_preds[idx] = lgb_p

        meta_X_oof = np.column_stack([ae_confs, ae_errors, lgb_confs, ae_preds, lgb_preds])
        meta_y_oof = y_train
        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "verbose": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }
        X_mt, X_mv, y_mt, y_mv = train_test_split(meta_X_oof, meta_y_oof, test_size=0.2, random_state=42)
        train_data = lgb.Dataset(X_mt, label=y_mt)
        val_data = lgb.Dataset(X_mv, label=y_mv)
        meta_booster = lgb.train(
            params, train_data, num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(10)],
        )

        calib_raws, calib_labels = [], []
        for i in range(len(X_val)):
            _, ae_c, ae_e = new_ae.predict_with_confidence(X_val[i])
            _, lgb_c, _ = new_lgb.predict_with_confidence(X_val[i])
            ae_pred_val = 1 if ae_c >= 0.5 else 0
            lgb_pred_val = 1 if lgb_c >= 0.5 else 0
            meta_in = np.array([[ae_c, ae_e, lgb_c, ae_pred_val, lgb_pred_val]], dtype=np.float32)
            raw = meta_booster.predict(meta_in, raw_score=True)[0]
            calib_raws.append(raw)
            calib_labels.append(y_val[i])
        calib_raws = np.array(calib_raws)
        calib_labels = np.array(calib_labels)
        if len(np.unique(calib_labels)) < 2:
            logger.warning("Calibration val set contains only one class -- skipping Platt calibration.")
            a, b = 1.0, 0.0
        else:
            from sklearn.linear_model import LogisticRegression
            calib_lr = LogisticRegression(C=1e10)
            calib_lr.fit(calib_raws.reshape(-1, 1), calib_labels)
            a = calib_lr.coef_[0][0]
            b = calib_lr.intercept_[0]

        with self._lock:
            self.autoencoder = new_ae
            self.lightgbm = new_lgb
            self.meta_model = meta_booster
            self._calib_a = a
            self._calib_b = b
        threading.Thread(target=self._init_shap_explainers, daemon=True).start()
        logger.info("Retrain complete: meta-model trained with OOF stacking and calibration.")
        return self

    def get_status(self):
        ae_status = self.autoencoder.get_status() if self.autoencoder else {}
        lgb_ready = self.lightgbm is not None
        meta_ready = self.meta_model is not None
        calib_ready = self._calib_a is not None and self._calib_b is not None
        scaler = self.autoencoder._scaler if self.autoencoder else None
        scaler_n = int(getattr(scaler, "n_samples_seen_", 0)) if scaler else 0
        return {
            "model_ready": ae_status.get("model_ready", False) and lgb_ready,
            "ae_ready": ae_status.get("model_ready", False),
            "lgb_ready": lgb_ready,
            "meta_ready": meta_ready,
            "calibrated": calib_ready,
            "ae_threshold": ae_status.get("threshold"),
            "feature_version": ae_status.get("feature_version", ""),
            "scaler_n": scaler_n,
            "ae_replay_size": 0,
            "prediction_error_count": ae_status.get("error_count", 0),
            "shap_available": SHAP_AVAILABLE and self._lgb_explainer is not None,
        }



# BackgroundRetrainer
class BackgroundRetrainer:
    """Feeds a retrain buffer from labeled feedback, monitors concept drift
    with ADWIN, and triggers automatic retraining on a background thread."""

    def __init__(self,
                 model: "HybridIDS",
                 min_samples: int = 500,
                 auto_interval: float = 60.0,
                 max_buffer_size: int = 100000,
                 max_consecutive_failures: int = 5,
                 feature_version: Optional[str] = None,
                 save_callback: Optional[Callable] = None,
                 model_update_callback: Optional[Callable] = None,
                 auto_retrain_disabled_cooldown_seconds: float = 1800.0):
        self.model = model
        self.min_samples = min_samples
        self.auto_interval = auto_interval
        self.max_buffer_size = max_buffer_size
        self.max_consecutive_failures = max_consecutive_failures
        self.feature_version = feature_version
        self.save_callback = save_callback
        self.model_update_callback = model_update_callback
        self.auto_retrain_disabled_cooldown_seconds = auto_retrain_disabled_cooldown_seconds
        self._buffer = collections.deque(maxlen=max_buffer_size)
        self.lock = threading.Lock()
        self._retrain_lock = threading.Lock()
        self.stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.drift_detector = ADWIN()
        self._drift_detected = False
        self._last_retrain_time = time.time()
        self._consecutive_failures = 0
        self._auto_retrain_disabled = False
        self._auto_retrain_disabled_at: Optional[float] = None
        self._drift_reset_cooldown = 60.0
        self._last_drift_reset = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("BackgroundRetrainer started.")

    def stop(self):
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            logger.info("BackgroundRetrainer stopped.")

    def add_samples(self, X, y):
        with self.lock:
            for xi, yi in zip(X, y):
                self._buffer.append((xi, yi))

    def buffer_size(self):
        with self.lock:
            return len(self._buffer)

    def get_status(self) -> dict:
        with self.lock:
            return {
                "auto_retrain_disabled": self._auto_retrain_disabled,
                "consecutive_failures": self._consecutive_failures,
                "buffer_size": len(self._buffer),
                "drift_detected": self._drift_detected,
            }

    def is_retrain_in_progress(self) -> bool:
        return self._retrain_lock.locked()

    def update_drift(self, error: float) -> bool:
        with self.lock:
            self.drift_detector.update(error)
            if self.drift_detector.drift_detected:
                self._drift_detected = True
            return self.drift_detector.drift_detected

    def detect_drift(self, prediction_error: float) -> bool:
        return self.update_drift(prediction_error)

    def retrain_now(self, force: bool = False, min_samples: Optional[int] = None) -> bool:
        if self.stop_event.is_set():
            return False
        with self._retrain_lock:
            if self.stop_event.is_set():
                return False
            if force:
                with self.lock:
                    self._auto_retrain_disabled = False
                    self._auto_retrain_disabled_at = None
            min_req = min_samples if min_samples is not None else self.min_samples
            with self.lock:
                total = len(self._buffer)
                if total < min_req and not force:
                    return False
                snapshot = list(self._buffer)
                X_np = np.array([x for x, _ in snapshot], dtype=np.float32)
                y_np = np.array([y for _, y in snapshot], dtype=np.int32)
                snap_count = len(X_np)
            try:
                logger.info("Starting retrain on %d samples.", snap_count)
                self.model.retrain(X_np, y_np, feature_version=self.feature_version)
                if self.model_update_callback:
                    self.model_update_callback(self.model)
                if self.save_callback:
                    self.save_callback()
                with self.lock:
                    drain = min(snap_count, len(self._buffer))
                    for _ in range(drain):
                        self._buffer.popleft()
                    self._consecutive_failures = 0
                    self._auto_retrain_disabled = False
                    self._auto_retrain_disabled_at = None
                    self._last_retrain_time = time.time()
                    now = time.time()
                    if now - self._last_drift_reset > self._drift_reset_cooldown:
                        self.drift_detector = ADWIN()
                        self._last_drift_reset = now
                        self._drift_detected = False
                logger.info("Retrain successful.")
                return True
            except Exception as e:
                logger.exception("Retrain failed: %s", e)
                with self.lock:
                    self._consecutive_failures += 1
                    drain_on_fail = max(1, snap_count // 10)
                    actual_drain = min(drain_on_fail, len(self._buffer))
                    for _ in range(actual_drain):
                        self._buffer.popleft()
                    if self._consecutive_failures >= self.max_consecutive_failures:
                        logger.error("Too many consecutive failures -- auto-retrain disabled.")
                        self._auto_retrain_disabled = True
                        self._auto_retrain_disabled_at = time.time()
                        self._consecutive_failures = 0
                return False

    def _worker_loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                enough = len(self._buffer) >= self.min_samples
                disabled = self._auto_retrain_disabled
                if disabled and self._auto_retrain_disabled_at is not None:
                    elapsed_since_disable = time.time() - self._auto_retrain_disabled_at
                    if elapsed_since_disable >= self.auto_retrain_disabled_cooldown_seconds:
                        self._auto_retrain_disabled = False
                        self._auto_retrain_disabled_at = None
                        self._consecutive_failures = 0
                        disabled = False
                elapsed = time.time() - self._last_retrain_time
                time_ok = elapsed >= self.auto_interval
                drift_detected = self._drift_detected
            if (enough and not disabled and time_ok) or drift_detected:
                self.retrain_now()
                self.stop_event.wait(1.0)
            else:
                self.stop_event.wait(5.0)


# Evaluation
def evaluate_model(model, X_test, y_test, batch_size: int = 1024):
    """Evaluate a model on a test set, returning accuracy, F1, AUC, etc."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    use_batch = hasattr(model, "predict_batch")

    if use_batch:
        preds_list, scores_list = [], []
        for start in range(0, len(X_test), batch_size):
            batch = X_test[start:start + batch_size]
            result = model.predict_batch(batch)
            if isinstance(result, tuple) and len(result) >= 2:
                p_batch, c_batch = result[0], result[1]
            else:
                p_batch, c_batch = result, np.zeros(len(batch))
            preds_list.append(p_batch)
            scores_list.append(c_batch)
        preds = np.concatenate(preds_list)
        scores = np.concatenate(scores_list)
    else:
        preds, scores = [], []
        for i in range(len(X_test)):
            result = model.predict_with_confidence(X_test[i])
            if isinstance(result, tuple):
                p, conf = result[0], result[1]
            else:
                p, conf = int(result), 0.0
            preds.append(p)
            scores.append(conf)
        preds = np.array(preds)
        scores = np.array(scores)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        roc_auc_val = roc_auc_score(y_test, scores)
    except ValueError:
        roc_auc_val = 0.0
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn + 1e-9)
    tpr_val = tp / (tp + fn + 1e-9)

    logger.info("MODEL EVALUATION ")
    logger.info("Accuracy : %.4f", acc)
    logger.info("Precision: %.4f", prec)
    logger.info("Recall   : %.4f", rec)
    logger.info("F1 Score : %.4f", f1)
    logger.info("ROC-AUC  : %.4f", roc_auc_val)
    logger.info("FPR      : %.4f", fpr_val)
    logger.info("TPR (DR) : %.4f", tpr_val)
    logger.info("Miss Rate: %.4f", 1 - tpr_val)
    logger.info("Method    : %s", "batch" if use_batch else "sample-by-sample")
    logger.info("....................")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc_val,
        "fpr": fpr_val,
        "tpr": tpr_val,
        "detection_rate": tpr_val,
        "miss_rate": float(1 - tpr_val),
    }
