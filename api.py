from fastapi import FastAPI
import pickle
import numpy as np
import os
import pandas as pd

DATA_DIR = "./data"

# ---------------------------
# 1. Alle Geste-DataFrames laden
# ---------------------------
gesture_data = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".pkl"):
        file_path = os.path.join(DATA_DIR, file)

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, pd.DataFrame):
            data["source_file"] = file
            gesture_data.append(data)

print(f"{len(gesture_data)} Dateien geladen.")

# ---------------------------
# 2. Trainierte Modelle laden
#    (du musst diese vorher gespeichert haben)
# ---------------------------
with open("./data/trained_hmm.pkl", "rb") as f:
    models = pickle.load(f)

print(f"{len(models)} HMM-Modelle geladen.")

app = FastAPI()

# ---------------------------
# 3. Prediction-Endpoint
# ---------------------------
@app.post("/predict")
async def predict(data: dict):
    # erwartet Feature-Vektor eines Frames
    features = np.array(data["features"])
    if features.ndim == 1:
        features = features.reshape(1, -1)


    scores = {}
    for gesture, model in models.items():
        try:
            score = model.score(features)
        except:
            score = -np.inf
        scores[gesture] = score

    predicted = max(scores, key=scores.get)

    return {"gesture": predicted}