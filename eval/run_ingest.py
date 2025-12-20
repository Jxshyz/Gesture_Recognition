from ingest_npy_sequences import ingest_npy_data

ingest_npy_data()

import sqlite3
from evaluation import evaluate

conn = sqlite3.connect("example.db")
print(evaluate(conn))