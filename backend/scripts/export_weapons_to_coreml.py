import sys
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
WEIGHTS_DIR = ROOT_DIR / "research" / "models" / "object_detection" / "weights"
SOURCE_PT = WEIGHTS_DIR / "best2.pt"


def main() -> int:
    if not SOURCE_PT.exists():
        print(f"ERROR: Source weights not found at {SOURCE_PT}")
        return 1

    print(f"Loading {SOURCE_PT}...")
    model = YOLO(str(SOURCE_PT))

    print("Exporting to CoreML (FP16, NMS embedded, imgsz=640)...")
    print("This takes 1-3 minutes on Apple Silicon.")
    output_path = model.export(
        format="coreml",
        half=True,
        imgsz=640,
        nms=True,
    )

    print(f"\nDone. Exported package: {output_path}")
    print("Next: just run `python main.py`. The worker will detect and use it automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

