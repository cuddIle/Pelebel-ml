import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
from pathlib import Path

from landmarks_utils import Landmark, create_image_pose_landmarker, extract_landmarks, compute_angles

CONSECUTIVE_DETECTIONS_REQUIRED = 5
DETECTION_INTERVAL = 0.5  # seconds
DISPLAY_LETTERS = ["A", "B", "C", "D"]

MODEL_PATH = Path("model/rf_classifier.joblib")
LABEL_ENCODER_PATH = Path("model/label_encoder.joblib")
POSE_MODEL_PATH = Path("model/pose_landmarker.task")

# Need all landmarks used by the angle definitions
ALLOWED_LANDMARKS = [
    Landmark.LEFT_SHOULDER,
    Landmark.RIGHT_SHOULDER,
    Landmark.LEFT_ELBOW,
    Landmark.RIGHT_ELBOW,
    Landmark.LEFT_WRIST,
    Landmark.RIGHT_WRIST,
    Landmark.LEFT_HIP,
    Landmark.RIGHT_HIP,
    Landmark.LEFT_KNEE,
    Landmark.RIGHT_KNEE,
    Landmark.LEFT_ANKLE,
    Landmark.RIGHT_ANKLE,
]


def _validate_paths() -> None:
    if not MODEL_PATH.exists():
        raise ValueError(f"Model not found: {MODEL_PATH}")
    if not LABEL_ENCODER_PATH.exists():
        raise ValueError(f"Label encoder not found: {LABEL_ENCODER_PATH}")
    if not POSE_MODEL_PATH.exists():
        raise ValueError(f"Pose model not found: {POSE_MODEL_PATH}")


def main():
    _validate_paths()

    print("Loading classifier...")
    classifier = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    print("Starting video capture...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    last_detection_time = 0.0
    displayed_letter = ""
    detection_history: list[str] = []
    letter_confidences: dict[str, float] = {}

    with create_image_pose_landmarker(model_path=POSE_MODEL_PATH, num_poses=1) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            landmarks_json = extract_landmarks(landmarker, mp_image, selected_landmarks=ALLOWED_LANDMARKS)
            if landmarks_json is None:
                detection_history.clear()
                displayed_letter = ""
                letter_confidences = {}
                cv2.putText(
                    frame, "No person detected", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
                )
                cv2.imshow("Letter Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Draw landmarks
            for lm in landmarks_json.values():
                x = int(lm["x"] * frame.shape[1])
                y = int(lm["y"] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            current_time = time.time()
            if current_time - last_detection_time >= DETECTION_INTERVAL:
                last_detection_time = current_time

                angles = compute_angles(landmarks_json)
                if angles is None:
                    detection_history.clear()
                    displayed_letter = ""
                    letter_confidences = {}
                    continue

                features = angles.reshape(1, -1)
                probabilities = classifier.predict_proba(features)[0]
                predicted_idx = int(np.argmax(probabilities))
                current_prediction = label_encoder.inverse_transform([predicted_idx])[0]

                # Store per-letter confidences for display letters
                letter_confidences = {}
                for lbl, prob in zip(label_encoder.classes_, probabilities):
                    if lbl in DISPLAY_LETTERS:
                        letter_confidences[lbl] = float(prob)

                # Track consecutive detections
                detection_history.append(current_prediction)
                if len(detection_history) > CONSECUTIVE_DETECTIONS_REQUIRED:
                    detection_history.pop(0)

                if (len(detection_history) == CONSECUTIVE_DETECTIONS_REQUIRED and
                        all(d == detection_history[0] for d in detection_history)):
                    displayed_letter = detection_history[0]

                # Console output
                print(f"\n--- Detection at {time.strftime('%H:%M:%S')} ---")
                print(f"  Current: {current_prediction} | Displayed: {displayed_letter} | History: {detection_history}")
                for lbl, prob in zip(label_encoder.classes_, probabilities):
                    bar = "█" * int(prob * 20)
                    print(f"  {lbl}: {prob:.1%} {bar}")

            # Draw prediction
            label_text = f"Letter: {displayed_letter}" if displayed_letter else "Letter: ---"
            cv2.putText(
                frame, label_text, (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4,
            )

            # Draw A/B/C/D confidences
            if letter_confidences:
                y_offset = 160
                for letter in DISPLAY_LETTERS:
                    conf = letter_confidences.get(letter, 0.0)
                    text = f"{letter}: {conf:.0%}"
                    color = (0, 255, 255) if conf >= 0.3 else (128, 128, 128)
                    cv2.putText(
                        frame, text, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3,
                    )
                    y_offset += 45

            cv2.putText(
                frame, "Press 'q' to quit",
                (50, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            cv2.imshow("Letter Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
