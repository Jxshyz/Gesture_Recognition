"""
Model loading and inference utilities.

This module provides helper functions to:

- Load a trained gesture classification model
- Load the corresponding label encoder
- Run inference on 189-dimensional feature vectors
- Decode encoded labels back to readable class names

Designed for runtime usage in live prediction pipelines.
"""
from __future__ import annotations
from typing import Any, Tuple
import numpy as np
import joblib


def load_model_and_encoder(
    model_path: str = "models/gesture_model.joblib",
    label_encoder_path: str = "models/label_encoder.joblib",
) -> Tuple[Any, Any]:
    """
    Load a trained model and its corresponding label encoder.

    Parameters:
        model_path (str):
            Path to the serialized model file (joblib format).

        label_encoder_path (str):
            Path to the serialized label encoder file.

    Returns:
        Tuple[Any, Any]:
            (model, label_encoder)
    """
    model = joblib.load(model_path)
    le = joblib.load(label_encoder_path)
    return model, le


def decode_label(y_enc: Any, le: Any) -> str:
    """
    Decode an encoded prediction label to its string representation.

    If decoding fails, the raw encoded value is returned as string.

    Parameters:
        y_enc (Any):
            Encoded prediction output from the model.

        le (Any):
            Label encoder used during training.

    Returns:
        str:
            Decoded class label.
    """
    try:
        return str(le.inverse_transform([y_enc])[0])
    except Exception:
        return str(y_enc)


def predict_feat189(model: Any, feat189: np.ndarray) -> Tuple[Any, float]:
    """
    Perform classification on a 189-dimensional feature vector.

    The input feature vector is reshaped to (1, 189) and passed to the model.

    If the model provides probability estimates (predict_proba),
    the maximum class probability is returned as confidence.
    Otherwise, confidence defaults to 1.0.

    Parameters:
        model (Any):
            Trained classification model.

        feat189 (np.ndarray):
            Feature vector of shape (189,).

    Returns:
        Tuple[Any, float]:
            - Encoded predicted label
            - Confidence score (0.0–1.0)
    """
    X = np.asarray(feat189, dtype=np.float32).reshape(1, -1)
    y = model.predict(X)[0]

    conf = 1.0
    if hasattr(model, "predict_proba"):
        conf = float(np.max(model.predict_proba(X)[0]))

    return y, conf
