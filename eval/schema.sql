-- =========================
-- Samples (eine Geste = ein Sample)
-- =========================
CREATE TABLE IF NOT EXISTS samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL
);

-- =========================
-- Embeddings (feste Vektoren pro Sample)
-- =========================
CREATE TABLE IF NOT EXISTS embeddings (
    sample_id INTEGER PRIMARY KEY,
    vector BLOB NOT NULL,
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
        ON DELETE CASCADE
);
