from similarity import find_similar


def precision_at_k(hits, k):
    """
    Compute Precision@K for a ranked retrieval result.

    Parameters:
        hits (list of int): Binary relevance list (1 = relevant, 0 = not relevant),
                            ordered by retrieval rank.
        k (int): Cutoff rank.

    Returns:
        float: Proportion of relevant items within the top-k results.
    """
    return sum(hits[:k]) / k


def average_precision(hits):
    """
    Compute Average Precision (AP) for a single ranked retrieval result.

    Average Precision is calculated as the mean of precision values
    at each rank where a relevant item occurs.

    Parameters:
        hits (list of int): Binary relevance list (1 = relevant, 0 = not relevant),
                            ordered by retrieval rank.

    Returns:
        float: Average Precision score. Returns 0 if no relevant items exist.
    """
    score = 0
    hit_count = 0
    for i, h in enumerate(hits):
        if h:
            hit_count += 1
            score += hit_count / (i + 1)
    return score / max(1, sum(hits))


def evaluate(conn, k=10):
    """
    Evaluate similarity-based retrieval performance using mAP and Precision@K.

    For each sample in the database:
        - Retrieve the top 50 most similar samples.
        - Build a binary relevance list based on matching labels.
        - Compute Average Precision (AP).
        - Compute Precision@K.

    Parameters:
        conn (sqlite3.Connection): Open SQLite database connection.
        k (int, optional): Rank cutoff for Precision@K. Default is 10.

    Returns:
        dict: Dictionary containing:
              - "mAP": Mean Average Precision across all queries.
              - "P@K": Mean Precision@K across all queries.
    """
    labels = {
        sid: label
        for sid, label in conn.execute("SELECT sample_id, label FROM samples")
    }

    all_ap = []
    all_p = []

    for q in labels.keys():
        results = find_similar(conn, q, top_k=50)
        hits = [1 if labels[sid] == labels[q] else 0 for sid, _ in results]
        all_ap.append(average_precision(hits))
        all_p.append(precision_at_k(hits, k))

    return {"mAP": sum(all_ap) / len(all_ap), f"P@{k}": sum(all_p) / len(all_p)}
