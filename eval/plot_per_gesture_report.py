import sqlite3
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from similarity import find_similar
from evaluation import average_precision, precision_at_k

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "example.db"


def compute_per_gesture(k=10):
    """
    Compute and print retrieval performance metrics per gesture class.

    For each sample in the database:
        - Retrieve the top 50 most similar samples.
        - Build a binary relevance list based on matching labels.
        - Compute Average Precision (AP).
        - Compute Precision@K.

    The metrics are aggregated per label (gesture class) and
    reported as:
        - mAP: Mean Average Precision per gesture
        - P@K: Mean Precision at rank K per gesture
        - N:   Number of query samples for that gesture

    Parameters:
        k (int, optional): Rank cutoff for Precision@K. Default is 10.

    Returns:
        None
    """
    conn = sqlite3.connect(DB_PATH)

    labels = {
        sid: label
        for sid, label in conn.execute("SELECT sample_id, label FROM samples")
    }

    ap_per_label = defaultdict(list)
    p_at_k_per_label = defaultdict(list)

    for query_id, query_label in labels.items():
        results = find_similar(conn, query_id, top_k=50)

        hits = [1 if labels[sid] == query_label else 0 for sid, _ in results]

        ap_per_label[query_label].append(average_precision(hits))
        p_at_k_per_label[query_label].append(precision_at_k(hits, k))

    conn.close()

    gestures = sorted(ap_per_label.keys())
    mAP = [np.mean(ap_per_label[g]) for g in gestures]
    p10 = [np.mean(p_at_k_per_label[g]) for g in gestures]

    return gestures, mAP, p10


def plot():
    """
    Visualize per-gesture retrieval performance as a grouped bar chart.

    The function retrieves gesture labels along with their
    corresponding mAP and P@10 scores using `compute_per_gesture()`.

    A side-by-side bar plot is created for each gesture:
        - mAP (Mean Average Precision)
        - P@10 (Precision at rank 10)

    The Y-axis represents the evaluation score (range 0–1).
    The X-axis lists gesture classes.

    Returns:
        None
    """
    gestures, mAP, p10 = compute_per_gesture()

    x = np.arange(len(gestures))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, mAP, width, label="mAP")
    plt.bar(x + width / 2, p10, width, label="P@10")

    plt.xticks(x, gestures, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-Gesture Evaluation")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot()
