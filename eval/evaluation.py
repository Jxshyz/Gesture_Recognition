
from similarity import find_similar

def precision_at_k(hits, k):
    return sum(hits[:k]) / k

def average_precision(hits):
    score = 0
    hit_count = 0
    for i, h in enumerate(hits):
        if h:
            hit_count += 1
            score += hit_count / (i + 1)
    return score / max(1, sum(hits))

def evaluate(conn, k=10):
    labels = {
        sid: label
        for sid, label in conn.execute("SELECT sample_id, label FROM samples")
    }

    all_ap = []
    all_p = []

    for q in labels.keys():
        results = find_similar(conn, q, top_k=50)
        hits = [
            1 if labels[sid] == labels[q] else 0
            for sid, _ in results
        ]
        all_ap.append(average_precision(hits))
        all_p.append(precision_at_k(hits, k))

    return {
        "mAP": sum(all_ap) / len(all_ap),
        f"P@{k}": sum(all_p) / len(all_p)
    }
