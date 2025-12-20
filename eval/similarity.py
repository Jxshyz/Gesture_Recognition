
import numpy as np
import pickle

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_embedding(conn, sample_id):
    cur = conn.execute(
        "SELECT vector FROM embeddings WHERE sample_id=?",
        (sample_id,)
    )
    return pickle.loads(cur.fetchone()[0])

def find_similar(conn, query_id, top_k=20):
    q_vec = load_embedding(conn, query_id)
    cur = conn.execute("SELECT sample_id, vector FROM embeddings")
    scores = []
    for sid, blob in cur.fetchall():
        if sid == query_id:
            continue
        vec = pickle.loads(blob)
        sim = cosine_similarity(q_vec, vec)
        scores.append((sid, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
