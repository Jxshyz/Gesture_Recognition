import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

def extract_numeric_features(df: pd.DataFrame) -> np.ndarray:
    lm_cols = [c for c in df.columns if c.startswith("lm_")]
    if not lm_cols:
        raise ValueError("no lm_-col found!")

    def parse_row(row):
        vals = []
        for v in row:
            if isinstance(v, (tuple, list)):
                vals.extend([float(x) if pd.notna(x) else 0.0 for x in v])
            else:
                vals.extend([0.0, 0.0, 0.0])
        return vals

    features = np.array([parse_row(df[lm_cols].iloc[i]) for i in range(len(df))], dtype=float)
    return features

def segment_gestures(df: pd.DataFrame):
    sequences, labels = [], []
    if "label_text" not in df.columns:
        raise ValueError("Missing 'label_text' column in DataFrame")

    current_label = None
    current_frames = []

    for _, row in df.iterrows():
        label = str(row["label_text"]).strip() if pd.notna(row["label_text"]) else None
        if label != current_label:
            if current_label and len(current_frames) > 0:
                segment_df = pd.DataFrame(current_frames)
                X = extract_numeric_features(segment_df)
                sequences.append(X)
                labels.append(current_label)
            current_label = label
            current_frames = []
        current_frames.append(row)

    if current_label and len(current_frames) > 0:
        segment_df = pd.DataFrame(current_frames)
        X = extract_numeric_features(segment_df)
        sequences.append(X)
        labels.append(current_label)

    return sequences, labels

def load_all_segments(data_dir: str):
    sequences, labels = [], []
    for file in tqdm(os.listdir(data_dir), desc="Loading gesture segments"):
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(data_dir, file)
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, pd.DataFrame):
                continue
            seqs, lbls = segment_gestures(data)
            sequences.extend(seqs)
            labels.extend(lbls)
        except Exception as e:
            print(f"⚠️ Error in {file}: {e}")
    return sequences, labels

def train_hmms(sequences, labels, n_states=5, n_iter=20):
    scaler = StandardScaler()
    scaler.fit(np.vstack(sequences))
    sequences = [scaler.transform(seq) for seq in sequences]

    gesture_groups = defaultdict(list)
    for seq, label in zip(sequences, labels):
        gesture_groups[label].append(seq)

    models = {}
    for gesture, seqs in tqdm(gesture_groups.items(), desc="Training all gestures"):
        lengths = [len(s) for s in seqs]
        X = np.vstack(seqs)

        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=1,
            verbose=False,
            init_params=""
        )

        for _ in tqdm(range(n_iter), desc=f"Baum–Welch {gesture}", leave=False):
            model.fit(X, lengths)

        models[gesture] = model

    return models, scaler

def predict_gesture(sequence, models, scaler):
    seq_scaled = scaler.transform(sequence)
    scores = {}
    for gesture, model in models.items():
        try:
            scores[gesture] = model.score(seq_scaled)
        except Exception:
            scores[gesture] = -np.inf
    return max(scores, key=scores.get)

if __name__ == "__main__":
    data_dir = "./data/"

    # loading and segmentation
    sequences, labels = load_all_segments(data_dir)
    print(f"\n→ {len(sequences)} gesture segments loaded from {len(set(labels))} unique gestures")

    # 2. Train/Test-Split per segment
    train_seqs, test_seqs, train_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"→ Training: {len(train_seqs)}, Test: {len(test_seqs)}")

    # 3. Training
    models, scaler = train_hmms(train_seqs, train_labels, n_states=5, n_iter=15)
    print("\nTraining finished!")

    # 4. Evaluation
    correct = 0
    for seq, lbl in tqdm(zip(test_seqs, test_labels), total=len(test_seqs), desc="Evaluating"):
        pred = predict_gesture(seq, models, scaler)
        if pred == lbl:
            correct += 1
    accuracy = correct / len(test_labels)
    print(f"\nTest accuracy: {accuracy:.2%}")

    # 5. Training accuracy
    correct_train = sum(predict_gesture(seq, models, scaler) == lbl for seq, lbl in zip(train_seqs, train_labels))
    print(f"Training accuracy: {correct_train / len(train_labels):.2%}")
