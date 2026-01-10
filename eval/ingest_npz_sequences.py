import sqlite3
import numpy as np
from pathlib import Path

from feature_extraction import extract_embedding, store_embedding

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "example.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"
DATA_ROOT = BASE_DIR.parent / "data"


def ingest_npz_data():
    print("[INFO] DB:", DB_PATH)
    print("[INFO] Data root:", DATA_ROOT)

    conn = sqlite3.connect(DB_PATH)

    # Schema laden
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    inserted = 0

    for npz_file in DATA_ROOT.rglob("*.npz"):
        data = np.load(npz_file)

        # 🔴 wir nutzen seq12
        if "seq12" not in data or "label" not in data:
            print(f"[WARN] Skip {npz_file.name} (missing keys)")
            continue

        seq = data["seq12"]          # (12, 63)
        label = str(data["label"])   # string

        if seq.shape != (12, 63):
            print(f"[WARN] Skip {npz_file.name} shape={seq.shape}")
            continue

        landmarks = seq.reshape(12, 21, 3)
        embedding = extract_embedding(landmarks)

        cur = conn.execute(
            "INSERT INTO samples (label) VALUES (?)",
            (label,)
        )
        sample_id = cur.lastrowid
        store_embedding(conn, sample_id, embedding)

        inserted += 1

    conn.commit()
    conn.close()

    print(f"[OK] Ingest abgeschlossen. Samples eingefügt: {inserted}")
