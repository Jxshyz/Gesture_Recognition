import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"


# --------------------------------------------------
# Load sequences from .npz
# --------------------------------------------------
def load_npz_sequences(data_root: Path):
    sequences = []
    labels = []

    for npz_file in data_root.rglob("*.npz"):
        data = np.load(npz_file)

        if "seq12" in data:
            seq = data["seq12"]  # (12, 63)
        elif "seq" in data:
            seq = data["seq"]
        else:
            continue
        # (T, 63)
        label = str(data["label"])

        if seq.ndim != 2 or seq.shape[1] != 63:
            continue

        sequences.append(seq.astype(np.float32))
        labels.append(label)

    return sequences, labels


# --------------------------------------------------
# Train HMMs (one per gesture)
# --------------------------------------------------
def train_hmms(sequences, labels, n_states=5, n_iter=20):
    # global scaling (important!)
    scaler = StandardScaler()
    scaler.fit(np.vstack(sequences))
    sequences = [scaler.transform(s) for s in sequences]

    grouped = defaultdict(list)
    for seq, lbl in zip(sequences, labels):
        grouped[lbl].append(seq)

    models = {}

    for label, seqs in tqdm(grouped.items(), desc="Training HMMs"):
        lengths = [len(s) for s in seqs]
        X = np.vstack(seqs)

        model = hmm.GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=n_iter, verbose=False
        )

        model.fit(X, lengths)
        models[label] = model

    return models, scaler


# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(sequence, models, scaler):
    seq = scaler.transform(sequence)
    scores = {}

    for label, model in models.items():
        try:
            scores[label] = model.score(seq)
        except Exception:
            scores[label] = -np.inf

    return max(scores, key=scores.get)


# --------------------------------------------------
# Main
# --------------------------------------------------
print("__name__ =", __name__)
if __name__ == "__main__":
    sequences, labels = load_npz_sequences(DATA_ROOT)

    print(f"Loaded {len(sequences)} sequences")
    print("Class distribution:", Counter(labels))

    # remove gestures with <2 samples
    counts = Counter(labels)
    valid = [i for i, l in enumerate(labels) if counts[l] >= 2]
    sequences = [sequences[i] for i in valid]
    labels = [labels[i] for i in valid]

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )

    models, scaler = train_hmms(X_train, y_train, n_states=5, n_iter=20)

    # evaluation
    from collections import defaultdict

    per_gesture_total = defaultdict(int)
    per_gesture_correct = defaultdict(int)

    for seq, true_label in zip(X_test, y_test):
        pred = predict(seq, models, scaler)
        per_gesture_total[true_label] += 1
        if pred == true_label:
            per_gesture_correct[true_label] += 1

    print("\nPer-Gesture Test Accuracy")
    print("=" * 35)

    for gesture in sorted(per_gesture_total.keys()):
        total = per_gesture_total[gesture]
        correct = per_gesture_correct[gesture]
        acc = correct / total
        print(f"{gesture:15s}: {acc:.2%}  ({correct}/{total})")

    overall_acc = sum(per_gesture_correct.values()) / sum(per_gesture_total.values())
    print("\nOverall Test Accuracy:", f"{overall_acc:.2%}")
