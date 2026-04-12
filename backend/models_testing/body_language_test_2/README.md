# Action Recognition Project (real trainable version)

This project replaces the fake browser-only classifier with a real temporal model that can be trained, saved, and used for live webcam inference.

## What is included

- A **real PyTorch temporal classifier** (`TemporalActionNet`) for 6 classes:
  - `neutral`
  - `punch`
  - `kick`
  - `push`
  - `wave`
  - `fall`
- A **bootstrap checkpoint** trained on synthetic motion sequences so the project can run immediately.
- A **public dataset downloader** wired to:
  - UTKinect skeleton joints + labels
  - UT-Interaction segmented videos
  - UP-Fall improved 3D skeleton archive mirrors
- A **dataset preparation pipeline** for UTKinect skeletons.
- A **webcam inference script** using MediaPipe Pose.
- A **second-stage training script** for real public data once downloaded and prepared.

## Important honesty note

The included checkpoint is a **synthetic bootstrap model**. It is much more real than the HTML you originally had because it is actually trained and saved, but it is **not the final public-data model yet**. The public-data training pipeline is included so you can convert this into a real dataset-based classifier on your machine.

I am being explicit about this because the original HTML used random/synthetic weights in JavaScript and rule heuristics, which are not a properly trained action classifier.

## Suggested workflow

### 1) Create the environment

```bash
python -m venv venv
# Windows
venv\Scriptsctivate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2) Run immediately with the bootstrap checkpoint

```bash
python infer_webcam.py
```

### 3) Download public data

```bash
python scripts/download_public_data.py
```

If you want to start smaller:

```bash
python scripts/download_public_data.py --only utkinect_joints utkinect_labels
```

### 4) Prepare UTKinect skeleton sequences

```bash
python scripts/prepare_public_datasets.py
```

### 5) Train a public-data checkpoint

```bash
python train_from_public_data.py
```

### 6) Use the new checkpoint

```bash
python infer_webcam.py --checkpoint models/public_data_temporal_action_net.pt
```

## Why this structure

- `UTKinect` gives you **skeleton joints** plus labels for `push` and `wave hands`, which are directly useful.
- `UT-Interaction` gives you `kicking`, `punching`, and `pushing` in realistic interaction videos.
- `UP-Fall` gives you **33-keypoint fall sequences** close to the MediaPipe representation used at inference time.

## Recommended next improvement

For the best real-world accuracy, add your own webcam clips for all 6 classes and fine-tune the public-data model with those same camera conditions. That usually reduces the domain gap a lot.
