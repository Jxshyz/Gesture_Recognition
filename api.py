from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import numpy as np
import pickle

app = FastAPI()

# Modell laden
with open("trained_gbm.pkl", "rb") as f:
    saved = pickle.load(f)
    gbm = saved["model"]
    scaler = saved["scaler"]


class FeatureInput(BaseModel):
    features: Union[List[float], List[List[float]]]


@app.post("/predict")
async def predict(data: FeatureInput):
    features = np.array(data.features)

    # Single Frame → 2D Array
    if features.ndim == 1:
        features = features.reshape(1, -1)
    elif features.ndim != 2:
        raise ValueError("Features must be 1D or 2D array")

    # Scale
    features_scaled = scaler.transform(features)

    # Prediction
    preds = gbm.predict(features_scaled)
    # Optional: Probabilities
    probs = gbm.predict_proba(features_scaled)

    return {"gesture": preds.tolist(), "probabilities": probs.tolist()}
