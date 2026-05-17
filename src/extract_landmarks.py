import json
from pathlib import Path

import mediapipe as mp

from landmarks_utils import create_image_pose_landmarker, extract_landmarks

SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png"]

DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("landmarks-dataset")
MODEL_PATH = Path("model/pose_landmarker.task")


def _validate_paths() -> None:
    if not DATASET_DIR.exists():
        raise ValueError(f"Dataset directory not found: {DATASET_DIR}")

    if not MODEL_PATH.exists():
        raise ValueError(f"Model not found: {MODEL_PATH}")

    if not OUTPUT_DIR.exists():
        raise ValueError(f"Output directory not found: {DATASET_DIR}")


def _validate_image(image_path: Path) -> bool:
    return image_path.suffix.lower() not in SUPPORTED_IMAGE_TYPES


def main():
    print("Starting landmark extraction")
    _validate_paths()

    with create_image_pose_landmarker(model_path=MODEL_PATH, num_poses=1) as landmarker:

        for case_dir in sorted(DATASET_DIR.iterdir()):
            print(f"Loading case dir {case_dir}")

            if not case_dir.is_dir():
                raise ValueError(f"Found non-directory: {case_dir}")

            output_letter_dir = OUTPUT_DIR / case_dir.name
            output_letter_dir.mkdir(exist_ok=True)

            for image_path in case_dir.iterdir():
                if _validate_image(image_path):
                    raise ValueError(f"  Skipping non-image file: {image_path.name}")

                mp_image = mp.Image.create_from_file(str(image_path))
                landmarks_json = extract_landmarks(landmarker, mp_image)
                if landmarks_json is None:
                    print(f"No landmarks detected in {image_path.name}")
                    continue

                output_path = output_letter_dir / f"{image_path.stem}.json"
                with open(output_path, "w") as f:
                    json.dump(landmarks_json, f)

            print(f"Finished extructing landmarks from {case_dir}")
        print("Finished landmark extraction")


if __name__ == "__main__":
    main()
