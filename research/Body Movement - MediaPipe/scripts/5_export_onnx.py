import torch
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models.system import TT2SkeletonSystem

def export():
    # 1. Find the latest checkpoint
    ckpt_paths = list(Path("mlruns").rglob("*.ckpt"))
    if not ckpt_paths:
        print("ERROR: No checkpoints found in mlruns/")
        return
        
    latest_ckpt = max(ckpt_paths, key=os.path.getctime)
    print(f"Loading {latest_ckpt}")

    # --- DEVICE FIX ---
    # Detect device (your RTX 4070 Ti SUPER will be 'cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting using device: {device}")

    # 2. Load the model and force it to the correct device
    model = TT2SkeletonSystem.load_from_checkpoint(latest_ckpt, map_location=device).eval()
    model.to(device) # Ensure the inner model layers are moved

    # 3. Create dummy input and move it to the SAME device as the model
    # (Batch, Channels, Frames, Joints)
    dummy_input = torch.zeros((1, 4, 15, 33)).to(device)

    out_path = "stgcn_pose.onnx"
    
    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            model.model,                 # Export the inner ST-GCN model
            dummy_input,
            out_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['skeleton_sequence'],
            output_names=['threat_logits'],
            dynamic_axes={
                'skeleton_sequence': {0: 'batch_size'}, 
                'threat_logits': {0: 'batch_size'}
            }
        )
        print(f"SUCCESS: Exported {out_path}")
    except Exception as e:
        print(f"FAILED to export: {e}")

if __name__ == "__main__":
    export()