import numpy as np
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

npz_files = list(DATA_ROOT.rglob("*.npz"))

print("Found npz files:", len(npz_files))
print("Example file:", npz_files[0])

data = np.load(npz_files[0])

print("Keys:", data.files)

for k in data.files:
    arr = data[k]
    print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
