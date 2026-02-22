import numpy as np
import pickle


def normalize_landmarks(landmarks):
    """
    Normalize landmark coordinates by translation and scale.

    The first landmark of each sample is used as the origin.
    All landmarks are translated so that this point becomes (0, 0, 0).
    The coordinates are then scaled by the mean L2 norm across all
    samples to achieve size normalization.

    Parameters:
        landmarks (np.ndarray): Array of shape (N, L, D),
                                where N = number of frames/samples,
                                L = number of landmarks,
                                D = coordinate dimensions.

    Returns:
        np.ndarray: Normalized landmark array with the same shape.
    """
    origin = landmarks[:, 0:1, :]
    landmarks = landmarks - origin
    scale = np.linalg.norm(landmarks, axis=(1, 2)).mean()
    if scale > 0:
        landmarks /= scale
    return landmarks


def extract_embedding(landmarks):
    """
    Generate a fixed-length embedding vector from landmark data.

    The landmarks are first normalized (translation and scale).
    Then the mean landmark configuration across all samples/frames
    is computed and flattened into a 1D feature vector.

    Parameters:
        landmarks (np.ndarray): Landmark array of shape (N, L, D).

    Returns:
        np.ndarray: 1D embedding vector representing the sample.
    """
    landmarks = normalize_landmarks(landmarks)
    mean_landmarks = landmarks.mean(axis=0)
    return mean_landmarks.flatten()


def store_embedding(conn, sample_id, vector):
    """
    Store an embedding vector in the database.

    The vector is converted to float32 and serialized using pickle
    before being stored as a BLOB in the 'embeddings' table.
    Existing entries with the same sample_id are replaced.

    Parameters:
        conn (sqlite3.Connection): Open SQLite database connection.
        sample_id (int): Identifier of the sample.
        vector (np.ndarray): Embedding vector to be stored.

    Returns:
        None
    """
    blob = pickle.dumps(vector.astype(np.float32))
    conn.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (sample_id, blob))
    conn.commit()
