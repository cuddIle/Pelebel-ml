from pathlib import Path

import joblib
import numpy as np
from skl2onnx import to_onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from landmarks_utils import compute_angles, load_landmarks

LANDMARKS_DIR = Path("landmarks-dataset")
MODEL_OUTPUT_PATH = Path("model/rf_classifier.joblib")
ONNX_OUTPUT_PATH = Path("model/rf_classifier.onnx")
LABEL_ENCODER_PATH = Path("model/label_encoder.joblib")


def load_dataset(
    debug: bool = False,
    balance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    data_per_class: dict[str, list[np.ndarray]] = {}
    total_per_class: dict[str, int] = {}
    skipped_per_class: dict[str, int] = {}

    for case_dir in sorted(LANDMARKS_DIR.iterdir()):
        if not case_dir.is_dir():
            raise ValueError(f"Found non-directory: {case_dir}")

        label = case_dir.name
        data_per_class[label] = []
        total_per_class[label] = 0
        skipped_per_class[label] = 0

        for json_path in case_dir.glob("*.json"):
            total_per_class[label] += 1
            landmarks = load_landmarks(json_path)
            angles = compute_angles(landmarks)
            if angles is None:
                skipped_per_class[label] += 1
                if debug:
                    print(f"  Skipped (angle computation failed): {json_path.name}")
                continue
            data_per_class[label].append(angles)

    if balance:
        min_samples = min(len(samples) for samples in data_per_class.values())
        print(f"\nBalancing dataset to {min_samples} samples per class")
        for label in data_per_class:
            if len(data_per_class[label]) > min_samples:
                indices = np.random.choice(len(data_per_class[label]), min_samples, replace=False)
                data_per_class[label] = [data_per_class[label][i] for i in indices]

    features_list = []
    labels_list = []
    for label, samples in data_per_class.items():
        features_list.extend(samples)
        labels_list.extend([label] * len(samples))

    if debug:
        print("\n=== Dataset Loading Debug ===")
        print("Samples per class (total -> kept" + (" -> balanced" if balance else "") + "):")
        for label in sorted(total_per_class.keys()):
            total = total_per_class[label]
            skipped = skipped_per_class[label]
            kept = total - skipped
            final = len(data_per_class[label])
            if balance:
                print(f"  {label}: {total} -> {kept} -> {final} (skipped {skipped})")
            else:
                print(f"  {label}: {total} -> {kept} (skipped {skipped})")

        features_arr = np.array(features_list)
        print(f"\nFeature statistics:")
        print(f"  Shape: {features_arr.shape}")
        print(f"  Min: {features_arr.min():.4f}, Max: {features_arr.max():.4f}")
        print(f"  Mean: {features_arr.mean():.4f}, Std: {features_arr.std():.4f}")

        nan_count = np.isnan(features_arr).sum()
        inf_count = np.isinf(features_arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  WARNING: Found {nan_count} NaN and {inf_count} Inf values!")

        print("=== End Debug ===\n")

    return np.array(features_list), np.array(labels_list)


def _validate_paths() -> None:
    if not LANDMARKS_DIR.exists():
        raise ValueError(f"Dataset directory not found: {LANDMARKS_DIR}")


def main():
    _validate_paths()

    print("Loading dataset (angle features)")
    features, labels = load_dataset(debug=True, balance=True)

    print(f"Loaded {len(features)} samples with {features.shape[1]} features (angles)")
    print(f"Classes: {sorted(set(labels))}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    print("\nTraining Random Forest classifier...")
    classifier = RandomForestClassifier(
        n_estimators=1000, min_samples_leaf=4, random_state=42, n_jobs=-1,
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    cv_scores = cross_val_score(classifier, features, y_encoded, cv=5)
    print(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print(f"\nSaving model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(classifier, MODEL_OUTPUT_PATH)

    print(f"Saving label encoder to {LABEL_ENCODER_PATH}...")
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print(f"Exporting ONNX model to {ONNX_OUTPUT_PATH}...")
    onx = to_onnx(
        classifier,
        X_train[:1].astype(np.float32),
        options={id(classifier): {"zipmap": False}},
    )
    ONNX_OUTPUT_PATH.write_bytes(onx.SerializeToString())

    print("\nDone!")


if __name__ == "__main__":
    main()
