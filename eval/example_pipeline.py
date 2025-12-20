import sqlite3
import numpy as np
from feature_extraction import extract_embedding, store_embedding
from evaluation import evaluate

conn = sqlite3.connect("example.db")

# 🔴 SCHEMA EINMALIG ERSTELLEN
with open("schema.sql") as f:
    conn.executescript(f.read())


# Beispiel: Fake-Landmarks erzeugen
for sample_id in range(1, 11):
    label = "gesture_a" if sample_id <= 5 else "gesture_b"
    conn.execute("INSERT OR REPLACE INTO samples VALUES (?, ?)", (sample_id, label))

    landmarks = np.random.rand(30, 21, 3)
    embedding = extract_embedding(landmarks)
    store_embedding(conn, sample_id, embedding)

conn.commit()

results = evaluate(conn)
print(results)
