"""
Export a YOLO model to a deployable inference format.

Default behavior exports the weapon-detection model (best2.pt) to CoreML
with FP16 at imgsz=640 — the configuration validated for our Apple
Silicon target. Arguments are exposed so the same script can be reused
for other models (face, future modules) and other formats (ONNX,
TensorRT, etc.) without editing source.

Usage examples
--------------
Default (weapons → CoreML FP16):
    python scripts/export_weapons_to_coreml.py

Custom weights:
    python scripts/export_weapons_to_coreml.py \\
        --weights research/models/object_detection/weights/best2.pt

Higher input resolution (must match training resolution to preserve accuracy):
    python scripts/export_weapons_to_coreml.py --imgsz 1280

Different format:
    python scripts/export_weapons_to_coreml.py --format onnx

Notes on the defaults
---------------------
- ``half=True`` exports in FP16. Apple Neural Engine runs natively in
  FP16; it is essentially a free speedup with no perceptible accuracy
  loss for object detection.
- ``imgsz=640`` matches the inference resolution used by the weapons
  worker. If the model was trained at a different resolution (e.g.
  1280), pass --imgsz to match training to recover full accuracy.
- ``nms`` defaults to True for parity with the original script. YOLOv10
  is end-to-end and ignores this flag (with a benign warning); older
  YOLO families respect it.
"""
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


# Repository-relative defaults so the script works regardless of
# checkout location, as long as the repo layout is preserved.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_WEIGHTS = (
    ROOT_DIR / "research" / "models" / "object_detection" / "weights" / "best2.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO model for accelerated inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to the source PyTorch weights (.pt).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="coreml",
        choices=["coreml", "onnx", "torchscript", "tflite", "engine"],
        help="Target export format.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size; must match training resolution for best accuracy.",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export with FP16 weights (recommended for Apple Neural Engine).",
    )
    parser.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bake NMS into the exported graph. YOLOv10 is end-to-end and ignores this.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        print(f"ERROR: Source weights not found at {args.weights}", file=sys.stderr)
        return 1

    print(f"Loading {args.weights}...")
    model = YOLO(str(args.weights))

    print(
        f"Exporting to {args.format} "
        f"(half={args.half}, imgsz={args.imgsz}, nms={args.nms})..."
    )
    print("This takes 1-3 minutes on Apple Silicon.")
    output_path = model.export(
        format=args.format,
        half=args.half,
        imgsz=args.imgsz,
        nms=args.nms,
    )

    print(f"\nDone. Exported package: {output_path}")
    print("Next: just run `python main.py`. The worker will detect and use it automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

