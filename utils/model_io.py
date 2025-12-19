from __future__ import annotations
from typing import Any, Tuple
import numpy as np
import joblib


def load_model_and_encoder(
    model_path: str = "models/gesture_model.joblib",
    label_encoder_path: str = "models/label_encoder.joblib",
) -> Tuple[Any, Any]:
    model = joblib.load(model_path)
    le = joblib.load(label_encoder_path)
    return model, le


def decode_label(y_enc: Any, le: Any) -> str:
    try:
        return str(le.inverse_transform([y_enc])[0])
    except Exception:
        return str(y_enc)


def predict_feat189(model: Any, feat189: np.ndarray) -> Tuple[Any, float]:
    """
    feat189: shape (189,)
    returns (y_enc, conf)
    """
    X = np.asarray(feat189, dtype=np.float32).reshape(1, -1)
    y = model.predict(X)[0]

    conf = 1.0
    if hasattr(model, "predict_proba"):
        conf = float(np.max(model.predict_proba(X)[0]))

    return y, conf
