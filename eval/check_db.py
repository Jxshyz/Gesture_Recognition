import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "example.db"

conn = sqlite3.connect(DB_PATH)

print("Tables:")
print(conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall())

print("\nSamples per label:")
print(conn.execute(
    "SELECT label, COUNT(*) FROM samples GROUP BY label"
).fetchall())
