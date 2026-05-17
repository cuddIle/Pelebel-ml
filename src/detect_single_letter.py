"""Detect letters from camera using individual letter models.

Captures from the webcam in real-time and checks each letter model to see
if that letter is being performed. Multiple letters can be detected
simultaneously. Detection runs in a background thread to keep video smooth.
"""

import threading
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from landmarks_utils import Landmark, compute_angles, create_image_pose_landmarker, extract_landmarks

MODELS_DIR = Path("model/single-letter-models")
POSE_MODEL_PATH = Path("model/pose_landmarker.task")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

DETECTION_THRESHOLD = 0.65  # Probability threshold for positive detection
DETECTION_INTERVAL = 1.0  # Seconds between detections (pose + classification)

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


def load_models() -> dict[str, any]:
    """Load all letter models that exist."""
    models = {}
    for letter in LETTERS:
        model_path = MODELS_DIR / f"{letter}.joblib"
        if model_path.exists():
            models[letter] = joblib.load(model_path)
    return models


def detect_letters(
    models: dict,
    features: np.ndarray,
    threshold: float = DETECTION_THRESHOLD,
) -> dict[str, float]:
    """Check each model and return detected letters with confidence."""
    detected = {}
    features_2d = features.reshape(1, -1)

    for letter, model in models.items():
        proba = model.predict_proba(features_2d)[0]
        confidence = proba[1] if len(proba) > 1 else proba[0]
        if confidence >= threshold:
            detected[letter] = confidence

    return detected


def detection_worker(
    models: dict,
    landmarker,
    frame_lock: threading.Lock,
    result_lock: threading.Lock,
    shared: dict,
):
    """Background thread that runs pose detection + classification."""
    while not shared["stop"]:
        # Grab the latest frame
        with frame_lock:
            frame = shared.get("frame")
        if frame is None:
            time.sleep(0.05)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        landmarks_json = extract_landmarks(
            landmarker, mp_image, selected_landmarks=ALLOWED_LANDMARKS
        )

        if landmarks_json is None:
            with result_lock:
                shared["detected"] = {}
                shared["landmarks"] = None
                shared["message"] = "No person detected"
        else:
            angles = compute_angles(landmarks_json)
            if angles is None:
                # Missing landmarks needed for angle computation
                with result_lock:
                    shared["detected"] = {}
                    shared["landmarks"] = landmarks_json
                    shared["message"] = "Not all landmarks visible"
            else:
                detected = detect_letters(models, angles)
                if detected:
                    sorted_letters = sorted(
                        detected.items(), key=lambda x: x[1], reverse=True
                    )
                    letters_str = ", ".join(
                        f"{l}:{c:.0%}" for l, c in sorted_letters
                    )
                    print(f"[{time.strftime('%H:%M:%S')}] Detected: {letters_str}")

                with result_lock:
                    shared["detected"] = detected
                    shared["landmarks"] = landmarks_json
                    shared["message"] = ""

        time.sleep(DETECTION_INTERVAL)


def main():
    if not POSE_MODEL_PATH.exists():
        print(f"Error: Pose model not found: {POSE_MODEL_PATH}")
        return

    print("Loading letter models...")
    models = load_models()
    if not models:
        print(f"Error: No models found in {MODELS_DIR}")
        print("Run train_single_letter_model.py first to train the models.")
        return
    print(f"Loaded {len(models)} models: {', '.join(sorted(models.keys()))}")

    print("\nStarting camera capture...")
    print(f"Detection threshold: {DETECTION_THRESHOLD:.0%}")
    print("Press 'q' to quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    frame_lock = threading.Lock()
    result_lock = threading.Lock()
    shared: dict = {
        "frame": None,
        "detected": {},
        "landmarks": None,
        "message": "",
        "stop": False,
    }

    with create_image_pose_landmarker(model_path=POSE_MODEL_PATH, num_poses=1) as landmarker:
        worker = threading.Thread(
            target=detection_worker,
            args=(models, landmarker, frame_lock, result_lock, shared),
            daemon=True,
        )
        worker.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)

            # Send frame to detection thread
            with frame_lock:
                shared["frame"] = frame.copy()

            display_frame = frame.copy()
            h, w = display_frame.shape[:2]

            # Read latest results
            with result_lock:
                detected_letters = dict(shared["detected"])
                last_landmarks = shared["landmarks"]
                message = shared["message"]

            # Draw landmarks
            if last_landmarks is not None:
                for lm in last_landmarks.values():
                    x = int(lm["x"] * w)
                    y = int(lm["y"] * h)
                    cv2.circle(display_frame, (x, y), 4, (0, 255, 0), -1)

            # Draw detected letters
            if detected_letters:
                sorted_letters = sorted(
                    detected_letters.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                # Each letter on its own line, stacked vertically centered
                n_letters = len(sorted_letters)
                line_height = 80
                total_height = n_letters * line_height
                start_y = (h - total_height) // 2 + 60

                for i, (letter, conf) in enumerate(sorted_letters):
                    text = f"{letter} {conf:.0%}"
                    font_scale = 3.0
                    thickness = 8
                    text_size = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = start_y + i * line_height

                    # Black outline
                    cv2.putText(
                        display_frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 4,
                    )
                    # Green text
                    cv2.putText(
                        display_frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness,
                    )
            elif message:
                text_size = cv2.getTextSize(
                    message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                )[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(
                    display_frame, message, (text_x, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                )
            else:
                cv2.putText(
                    display_frame, "---",
                    (w // 2 - 80, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (128, 128, 128), 8,
                )

            cv2.putText(
                display_frame, "Press 'q' to quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

            cv2.imshow("Letter Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    shared["stop"] = True
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
