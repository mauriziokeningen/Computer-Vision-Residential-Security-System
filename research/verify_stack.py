import mlflow
import time
import torch

# 1. Direct the logic to the local tracking server
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("System_Verification")

with mlflow.start_run(run_name="Handshake_Check"):
    print("Initiating SOTA Handshake...")
    
    # Log hardware specs
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    mlflow.log_param("compute_engine", gpu_name)
    mlflow.log_param("architect", "USER")
    
    # Log a dummy training curve
    for i in range(10):
        integrity_score = (i + 1) * 10
        mlflow.log_metric("handshake_integrity_pct", integrity_score, step=i)
        time.sleep(0.1)
    
    print(f"Logic Check: Stack verified on {gpu_name}.")
    print("Check your laptop browser at http://127.0.0.1:8080")