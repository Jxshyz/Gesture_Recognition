
import numpy as np
import pickle
import sqlite3

def normalize_landmarks(landmarks):
    origin = landmarks[:, 0:1, :]
    landmarks = landmarks - origin
    scale = np.linalg.norm(landmarks, axis=(1,2)).mean()
    if scale > 0:
        landmarks /= scale
    return landmarks

def extract_embedding(landmarks):
    landmarks = normalize_landmarks(landmarks)
    mean_landmarks = landmarks.mean(axis=0)
    return mean_landmarks.flatten()

def store_embedding(conn, sample_id, vector):
    blob = pickle.dumps(vector.astype(np.float32))
    conn.execute(
        "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
        (sample_id, blob)
    )
    conn.commit()
