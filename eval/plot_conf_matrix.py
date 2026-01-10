import sqlite3
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from similarity import find_similar

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "example.db"


def compute_confusion():
    conn = sqlite3.connect(DB_PATH)

    labels = {
        sid: label
        for sid, label in conn.execute("SELECT sample_id, label FROM samples")
    }

    label_names = sorted(set(labels.values()))
    idx = {l: i for i, l in enumerate(label_names)}

    mat = np.zeros((len(label_names), len(label_names)), dtype=np.float32)

    for qid, true_label in labels.items():
        res = find_similar(conn, qid, top_k=1)
        if not res:
            continue
        pred_id, _ = res[0]
        pred_label = labels[pred_id]
        mat[idx[true_label], idx[pred_label]] += 1

    # Normalize rows
    mat = mat / mat.sum(axis=1, keepdims=True)

    conn.close()
    return label_names, mat


def plot():
    labels, mat = compute_confusion()

    plt.figure(figsize=(8, 7))
    plt.imshow(mat)
    plt.colorbar(label="Probability")

    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Similarity Confusion Matrix (Top-1)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot()
