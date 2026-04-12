from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

PUBLIC_ASSETS = {
    # UTKinect official skeletons + labels
    "utkinect_joints": "https://cvrc.ece.utexas.edu/KinectDatasets/joints.zip",
    "utkinect_labels": "https://cvrc.ece.utexas.edu/KinectDatasets/actionLabel.txt",
    # UT-Interaction official segmented sets
    "utinteraction_set1_segmented": "https://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set1.zip",
    "utinteraction_set2_segmented": "https://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set2.zip",
    "utinteraction_ground_truth": "https://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_labels_110912.xls",
    # UP-Fall improved 3D skeletons repo mirrors
    "upfall_subject1": "https://raw.githubusercontent.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset/main/SUBJECT1.zip",
    "upfall_subject2": "https://raw.githubusercontent.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset/main/SUBJECT2.zip",
    "upfall_subject3": "https://raw.githubusercontent.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset/main/SUBJECT3.zip",
    "upfall_subject4": "https://raw.githubusercontent.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset/main/SUBJECT4.zip",
    "upfall_subject5": "https://raw.githubusercontent.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset/main/SUBJECT5.zip",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public action-recognition datasets used by this project.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/public_datasets"))
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional asset keys to download. Available keys: " + ", ".join(PUBLIC_ASSETS.keys()),
    )
    args = parser.parse_args()

    selected = PUBLIC_ASSETS if not args.only else {k: PUBLIC_ASSETS[k] for k in args.only}
    for key, url in selected.items():
        suffix = url.split("/")[-1]
        dest = args.output_dir / suffix
        print(f"Downloading {key} -> {dest}")
        download_file(url, dest)


if __name__ == "__main__":
    main()
