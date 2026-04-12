@echo off
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python scripts\download_public_data.py --only utkinect_joints utkinect_labels
python scripts\prepare_public_datasets.py
python train_from_public_data.py
pause
