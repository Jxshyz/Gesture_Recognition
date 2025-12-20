import sqlite3
import numpy as np
from pathlib import Path

from feature_extraction import extract_embedding, store_embedding

DB_PATH = "example.db"
DATA_ROOT = Path("./data_raw")

def ingest_npy_data():
    conn = sqlite3.connect(DB_PATH)

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    SCHEMA_PATH = BASE_DIR / "schema.sql"

    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    sample_id = 0

    # Ordnerstruktur: data_raw/<gesture>/<person>/*.npy
    for gesture_dir in DATA_ROOT.iterdir():
        if not gesture_dir.is_dir():
            continue
        label = gesture_dir.name

        for person_dir in gesture_dir.iterdir():
            if not person_dir.is_dir():
                continue

            for npy_file in person_dir.glob("*.npy"):
                seq = np.load(npy_file)  # (frames, 63)

                if seq.ndim != 2:
                    continue

                # zurück zu (frames, 21, 3)
                landmarks = seq.reshape(seq.shape[0], 21, 3)

                embedding = extract_embedding(landmarks)

                sample_id += 1
                cur = conn.execute(
                    "INSERT INTO samples (label) VALUES (?)",
                    (label,)
                )
                sample_id = cur.lastrowid
                store_embedding(conn, sample_id, embedding)


    conn.commit()
    conn.close()
