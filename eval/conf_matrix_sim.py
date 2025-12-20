import sqlite3
from collections import defaultdict

from similarity import find_similar


def confusion_matrix_similarity(db_path="example.db"):
    conn = sqlite3.connect(db_path)

    labels = {
        sid: label
        for sid, label in conn.execute("SELECT sample_id, label FROM samples")
    }

    # confusion[true][pred] += 1
    confusion = defaultdict(lambda: defaultdict(int))

    for query_id, true_label in labels.items():
        results = find_similar(conn, query_id, top_k=1)

        if not results:
            continue

        pred_id, _ = results[0]
        pred_label = labels[pred_id]

        confusion[true_label][pred_label] += 1

    all_labels = sorted(confusion.keys())

    print("\nSimilarity Confusion Matrix (Top-1)")
    print("=" * 60)

    header = "true\\pred".ljust(15) + "".join(l.ljust(15) for l in all_labels)
    print(header)
    print("-" * len(header))

    for true_label in all_labels:
        row = true_label.ljust(15)
        total = sum(confusion[true_label].values())

        for pred_label in all_labels:
            count = confusion[true_label].get(pred_label, 0)
            val = f"{count/total:.2f}" if total > 0 else "0.00"
            row += val.ljust(15)

        print(row)

    conn.close()

if __name__ == "__main__":
    confusion_matrix_similarity()
