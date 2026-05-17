"""Train individual binary classifiers for each letter A-Z.

Each model detects whether a pose matches a specific letter or not.
Uses all other letters as negative samples.
"""

from pathlib import Path

import joblib
import numpy as np
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from joblib_to_json import export_forest
from landmarks_utils import compute_angles, load_landmarks

LANDMARKS_DIR = Path("landmarks-dataset")
OUTPUT_DIR = Path("model/single-letter-models")
ONNX_OUTPUT_DIR = Path("model/single-letter-models-onnx")
JSON_OUTPUT_DIR = Path("model/single-letter-models-json")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_all_samples() -> dict[str, list[np.ndarray]]:
    """Load all angle features grouped by label."""
    samples_by_label: dict[str, list[np.ndarray]] = {}

    for case_dir in sorted(LANDMARKS_DIR.iterdir()):
        if not case_dir.is_dir():
            continue

        label = case_dir.name
        samples_by_label[label] = []

        for json_path in case_dir.glob("*.json"):
            landmarks = load_landmarks(json_path)
            angles = compute_angles(landmarks)
            if angles is not None:
                samples_by_label[label].append(angles)

    return samples_by_label


def train_letter_model(
    target_letter: str,
    samples_by_label: dict[str, list[np.ndarray]],
) -> tuple[MLPClassifier, StandardScaler, dict]:
    """Train a binary classifier for a single letter.

    Args:
        target_letter: The letter to detect (e.g., "A")
        samples_by_label: Dictionary mapping labels to feature arrays

    Returns:
        Trained classifier and metrics dictionary
    """
    positive_samples = samples_by_label.get(target_letter, [])
    if not positive_samples:
        raise ValueError(f"No samples found for letter {target_letter}")

    # Sample negatives evenly from each other class so no single class
    # (like NOT_A_LETTER) dominates. Match total negatives to positives.
    n_positive = len(positive_samples)
    other_labels = [l for l in samples_by_label if l != target_letter]
    per_class = max(1, n_positive // len(other_labels))

    rng = np.random.RandomState(42)
    negative_samples = []
    for label in other_labels:
        pool = samples_by_label[label]
        n_take = min(per_class, len(pool))
        indices = rng.choice(len(pool), n_take, replace=False)
        negative_samples.extend(pool[i] for i in indices)

    # Build feature matrix and labels (roughly 1:1 ratio)
    X = np.array(positive_samples + negative_samples)
    y = np.array([1] * len(positive_samples) + [0] * len(negative_samples))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifier = RandomForestClassifier(
        n_estimators=1000,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)

    metrics = {
        "letter": target_letter,
        "positive_samples": len(positive_samples),
        "negative_samples": len(negative_samples),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return classifier, metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all samples...")
    samples_by_label = load_all_samples()

    print("\nSample counts:")
    for label in sorted(samples_by_label.keys()):
        print(f"  {label}: {len(samples_by_label[label])}")

    print(f"\nTraining {len(LETTERS)} letter models...")
    print("-" * 60)

    all_metrics = []
    for letter in LETTERS:
        if letter not in samples_by_label or not samples_by_label[letter]:
            print(f"  {letter}: SKIPPED (no samples)")
            continue

        classifier, metrics = train_letter_model(letter, samples_by_label)
        all_metrics.append(metrics)

        output_path = OUTPUT_DIR / f"{letter}.joblib"
        joblib.dump(classifier, output_path)

        sample = np.array([samples_by_label[letter][0]], dtype=np.float32)
        onx = to_onnx(
            classifier, sample, options={id(classifier): {"zipmap": False}},
        )
        onnx_path = ONNX_OUTPUT_DIR / f"{letter}.onnx"
        onnx_path.write_bytes(onx.SerializeToString())

        json_path = JSON_OUTPUT_DIR / f"{letter}.json"
        export_forest(output_path, json_path)

        print(
            f"  {letter}: f1={metrics['f1']:.3f} "
            f"prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} "
            f"(+:{metrics['positive_samples']}, -:{metrics['negative_samples']})"
        )

    print("-" * 60)

    # Summary
    f1_scores = [m["f1"] for m in all_metrics]
    print(f"\nSummary:")
    print(f"  Models trained: {len(all_metrics)}")
    print(f"  Mean F1: {np.mean(f1_scores):.3f}")
    print(f"  Min F1:  {np.min(f1_scores):.3f} ({all_metrics[np.argmin(f1_scores)]['letter']})")
    print(f"  Max F1:  {np.max(f1_scores):.3f} ({all_metrics[np.argmax(f1_scores)]['letter']})")
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print(f"ONNX models saved to: {ONNX_OUTPUT_DIR}")
    print(f"JSON models saved to: {JSON_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
