import numpy as np
import pickle


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.

    Cosine similarity is defined as the dot product of the vectors
    divided by the product of their L2 norms.

    Parameters:
        a (np.ndarray): First embedding vector.
        b (np.ndarray): Second embedding vector.

    Returns:
        float: Cosine similarity score in the range [-1, 1].
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_embedding(conn, sample_id):
    """
    Load a stored embedding vector from the database.

    The embedding is retrieved from the 'embeddings' table,
    deserialized from a BLOB using pickle, and returned
    as a NumPy array.

    Parameters:
        conn (sqlite3.Connection): Open SQLite database connection.
        sample_id (int): Identifier of the sample.

    Returns:
        np.ndarray: Deserialized embedding vector.
    """
    cur = conn.execute("SELECT vector FROM embeddings WHERE sample_id=?", (sample_id,))
    return pickle.loads(cur.fetchone()[0])


def find_similar(conn, query_id, top_k=20):
    """
    Retrieve the top-K most similar samples for a given query sample.

    The function:
        - Loads the query embedding.
        - Computes cosine similarity against all other embeddings.
        - Sorts results by similarity in descending order.
        - Returns the top_k highest scoring samples.

    The query sample itself is excluded from the results.

    Parameters:
        conn (sqlite3.Connection): Open SQLite database connection.
        query_id (int): Identifier of the query sample.
        top_k (int, optional): Number of most similar samples to return.
                               Default is 20.

    Returns:
        list[tuple[int, float]]: List of (sample_id, similarity_score)
                                 sorted by descending similarity.
    """
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
