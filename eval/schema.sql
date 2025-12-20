
CREATE TABLE IF NOT EXISTS samples (
    sample_id INTEGER PRIMARY KEY,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS landmarks (
    sample_id INTEGER,
    frame_index INTEGER,
    landmark_index INTEGER,
    x REAL,
    y REAL,
    z REAL,
    PRIMARY KEY (sample_id, frame_index, landmark_index)
);

CREATE TABLE IF NOT EXISTS embeddings (
    sample_id INTEGER PRIMARY KEY,
    vector BLOB
);
