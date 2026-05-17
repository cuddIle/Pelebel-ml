import random
import shutil
from pathlib import Path

DATASET_DIR = Path("dataset")
TARGET_DIR = DATASET_DIR / "NOT_A_LETTER"
SAMPLES_PER_LETTER = 5
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42


def main() -> None:
    random.seed(SEED)
    TARGET_DIR.mkdir(exist_ok=True)

    copied = 0
    for letter_dir in sorted(DATASET_DIR.iterdir()):
        if not letter_dir.is_dir() or letter_dir == TARGET_DIR:
            continue

        images = [p for p in letter_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if not images:
            print(f"  {letter_dir.name}: no images, skipping")
            continue

        sample = random.sample(images, min(SAMPLES_PER_LETTER, len(images)))
        for src in sample:
            dst = TARGET_DIR / f"{letter_dir.name}_{src.name}"
            shutil.copy2(src, dst)
            copied += 1

        print(f"  {letter_dir.name}: copied {len(sample)} images")

    print(f"\nDone. Copied {copied} images into {TARGET_DIR}")


if __name__ == "__main__":
    main()
