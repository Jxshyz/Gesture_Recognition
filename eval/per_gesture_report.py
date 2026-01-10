import sqlite3
from pathlib import Path
from collections import defaultdict

from similarity import find_similar
from evaluation import average_precision, precision_at_k

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "example.db"


def per_gesture_report(k=10):
    conn = sqlite3.connect(DB_PATH)


    labels = {
        sid: label
        for sid, label in conn.execute("SELECT sample_id, label FROM samples")
    }

    ap_per_label = defaultdict(list)
    p_at_k_per_label = defaultdict(list)

    for query_id, query_label in labels.items():
        results = find_similar(conn, query_id, top_k=50)

        hits = [
            1 if labels[sid] == query_label else 0
            for sid, _ in results
        ]

        ap = average_precision(hits)
        p_k = precision_at_k(hits, k)

        ap_per_label[query_label].append(ap)
        p_at_k_per_label[query_label].append(p_k)

    print("\nPer-Gesture Evaluation")
    print("=" * 40)
    print(f"{'Gesture':15s}  {'mAP':>6s}  {'P@'+str(k):>6s}  {'N':>4s}")
    print("-" * 40)

    """
    mAP: Mean Average Precision
    P@10: Precision of top ten similarities
    N: Number of samples
    """

    for label in sorted(ap_per_label.keys()):
        mean_ap = sum(ap_per_label[label]) / len(ap_per_label[label])
        mean_p = sum(p_at_k_per_label[label]) / len(p_at_k_per_label[label])
        n = len(ap_per_label[label])

        print(f"{label:15s}  {mean_ap:6.3f}  {mean_p:6.3f}  {n:4d}")

    conn.close()

if __name__ == "__main__":
    per_gesture_report()
