import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score
import sys
import os
from pathlib import Path
import mlflow # <--- NEW IMPORT

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models.system import TT2SkeletonSystem

def evaluate():
    # 1. Automatically find the newest checkpoint
    ckpt_paths = list(Path("mlruns").rglob("*.ckpt"))
    if not ckpt_paths:
        print("ERROR: Could not find any .ckpt files in the mlruns directory.")
        return
    
    latest_ckpt = max(ckpt_paths, key=os.path.getctime)
    print(f"Loading latest MLflow checkpoint: {latest_ckpt}")
    
    # 2. Extract the MLflow Run ID from the folder structure
    # Path format: mlruns / <experiment_id> / <run_id> / artifacts / ...
    run_id = latest_ckpt.parts[2] 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TT2SkeletonSystem.load_from_checkpoint(latest_ckpt)
    model.to(device).eval()

    # 3. Load Processed Val Data
    X_val = np.load("data/processed/X_val.npy")
    y_true = np.load("data/processed/y_val.npy")

    # 4. Batch Inference 
    y_probs = []
    with torch.no_grad():
        for i in range(0, len(X_val), 64):
            batch = torch.from_numpy(X_val[i:i+64]).float().permute(0, 3, 1, 2).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_probs.extend(probs.cpu().numpy())

    y_probs = np.array(y_probs)
    y_preds = (y_probs > 0.5).astype(int)

    # 5. Connect to MLflow and "Resume" the Run
    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_id=run_id):
        
        # Log the final hard numbers to the UI
        print(f"\n--- SOTA AUDIT RESULTS ---")
        acc = accuracy_score(y_true, y_preds)
        rec = recall_score(y_true, y_preds)
        prec = precision_score(y_true, y_preds)
        f1 = f1_score(y_true, y_preds)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Recall:    {rec:.4f}  <-- Target: > 0.85")
        print(f"Precision: {prec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        mlflow.log_metrics({
            "final_audit_acc": acc,
            "final_audit_recall": rec,
            "final_audit_precision": prec,
            "final_audit_f1": f1
        })

        # 6. Generate & Upload Confusion Matrix
        cm = confusion_matrix(y_true, y_preds)
        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral', 'Threat'], yticklabels=['Neutral', 'Threat'])
        plt.title('ST-GCN Threat Detection Matrix')
        plt.ylabel('Actual Truth')
        plt.xlabel('Model Prediction')
        mlflow.log_figure(fig_cm, "evaluation_plots/confusion_matrix.png") # <--- UPLOADS TO MLFLOW

        # 7. Generate & Upload PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        fig_pr = plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='purple', lw=2)
        plt.xlabel('Recall (Catching the threat)')
        plt.ylabel('Precision (Avoiding false alarms)')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        mlflow.log_figure(fig_pr, "evaluation_plots/pr_curve.png") # <--- UPLOADS TO MLFLOW

        print(f"\nSUCCESS! Images successfully attached to MLflow Run ID: {run_id}")

if __name__ == "__main__":
    evaluate()