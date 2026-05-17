import json
from enum import IntEnum
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions


class Landmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def dump_landmarks(landmarks: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(landmarks, f)


def load_landmarks(path: Path, allowed_landmarks: list[Landmark] | None = None) -> dict:
    with open(path) as f:
        data = json.load(f)
    if allowed_landmarks is None:
        return data
    allowed_keys = {lm.name.lower() for lm in allowed_landmarks}
    return {k: v for k, v in data.items() if k in allowed_keys}


def landmarks_to_array(landmarks: dict, default_value: float = 0.0) -> np.ndarray:
    result = []
    for lm in Landmark:
        key = lm.name.lower()
        if key in landmarks:
            result.append([landmarks[key]["x"], landmarks[key]["y"]])
        else:
            result.append([default_value, default_value])
    return np.array(result)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray | None:
    """
    Normalize landmarks for translation and scale invariance.

    Args:
        landmarks: Array of shape (33, 2) containing raw MediaPipe pose landmarks (x, y only).

    Returns:
        Normalized landmarks array of shape (33, 2), or None if normalization fails
        (e.g., shoulders are at the same position).

    Normalization steps:
        1. Translation: Center landmarks by subtracting the mid-hip point
           (average of LEFT_HIP and RIGHT_HIP).
        2. Scale: Divide all coordinates by the shoulder distance
           (Euclidean distance between LEFT_SHOULDER and RIGHT_SHOULDER).
    """
    landmarks = landmarks.copy()

    # Calculate mid-hip point (center of LEFT_HIP and RIGHT_HIP)
    left_hip = landmarks[Landmark.LEFT_HIP]
    right_hip = landmarks[Landmark.RIGHT_HIP]
    mid_hip = (left_hip + right_hip) / 2

    # Center all landmarks by subtracting mid-hip
    landmarks -= mid_hip

    # Calculate shoulder distance for scale normalization
    left_shoulder = landmarks[Landmark.LEFT_SHOULDER]
    right_shoulder = landmarks[Landmark.RIGHT_SHOULDER]
    shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)

    # Avoid division by zero
    if shoulder_distance < 1e-6:
        return None

    # Scale all landmarks by shoulder distance
    landmarks /= shoulder_distance

    return landmarks


ANGLE_DEFINITIONS: list[tuple[str, Landmark, Landmark, Landmark]] = [
    ("left_elbow", Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    ("right_elbow", Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
    ("left_shoulder", Landmark.LEFT_HIP, Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
    ("right_shoulder", Landmark.RIGHT_HIP, Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
    ("left_hip", Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
    ("right_hip", Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    ("left_knee", Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    ("right_knee", Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    ("left_armpit", Landmark.RIGHT_SHOULDER, Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
    ("right_armpit", Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
    ("left_side_bend", Landmark.LEFT_ANKLE, Landmark.LEFT_HIP, Landmark.LEFT_SHOULDER),
    ("right_side_bend", Landmark.RIGHT_ANKLE, Landmark.RIGHT_HIP, Landmark.RIGHT_SHOULDER),
    ("left_leg_chain", Landmark.LEFT_ANKLE, Landmark.LEFT_KNEE, Landmark.LEFT_HIP),
    ("right_leg_chain", Landmark.RIGHT_ANKLE, Landmark.RIGHT_KNEE, Landmark.RIGHT_HIP),
]


def _get_point(landmarks: dict, lm: Landmark) -> np.ndarray | None:
    key = lm.name.lower()
    if key not in landmarks:
        return None
    return np.array([landmarks[key]["x"], landmarks[key]["y"]])


def compute_angles(landmarks: dict) -> np.ndarray | None:
    """
    Compute joint angles from landmarks dict.

    Returns array of shape (10,) with angles in degrees, or None if
    required landmarks are missing.
    """
    angles = []
    for _name, p1_lm, vertex_lm, p2_lm in ANGLE_DEFINITIONS:
        p1 = _get_point(landmarks, p1_lm)
        vertex = _get_point(landmarks, vertex_lm)
        p2 = _get_point(landmarks, p2_lm)
        if p1 is None or vertex is None or p2 is None:
            return None
        v1 = p1 - vertex
        v2 = p2 - vertex
        cos_angle = np.dot(v1, v2)
        sin_angle = v1[0] * v2[1] - v1[1] * v2[0]
        angle = np.degrees(np.arctan2(sin_angle, cos_angle))
        angles.append(angle)
    return np.array(angles)


def create_image_pose_landmarker(model_path: Path, num_poses: int = 1):
    return PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=num_poses,
    ))


def extract_landmarks(landmarker, image: mp.Image, selected_landmarks: list[Landmark] | None = None) -> dict | None:
    result = landmarker.detect(image)

    if not result.pose_landmarks:
        return None

    pose_landmarks = result.pose_landmarks[0]
    extracted = {}
    for landmark in Landmark:
        if selected_landmarks is not None and landmark not in selected_landmarks:
            continue
        if landmark.value < len(pose_landmarks):
            lm = pose_landmarks[landmark.value]
            extracted[landmark.name.lower()] = {"x": lm.x, "y": lm.y}

    return extracted if extracted else None
