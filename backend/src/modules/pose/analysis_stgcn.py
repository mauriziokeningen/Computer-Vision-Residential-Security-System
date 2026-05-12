import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import os

logger = logging.getLogger("PoseAnalysis")

# --- ST-GCN ARCHITECTURE (Copied from Research) ---
class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1, inplace=True)
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x)
        x = torch.matmul(x, A) 
        x = self.tcn(x) + res
        return self.relu(x)

class TT2SkeletonClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=2): 
        super().__init__()
        A = torch.zeros((17, 17))
        coco_pairs = [
            (0,1), (0,2), (1,3), (2,4), (0,5), (0,6), (5,7), 
            (7,9), (6,8), (8,10), (5,11), (6,12), (11,13), 
            (13,15), (12,14), (14,16), (11,12)
        ]
        for i in range(17): A[i, i] = 1.0
        for i, j in coco_pairs:
            A[i, j] = 1.0
            A[j, i] = 1.0
            
        D = torch.diag(torch.sum(A, dim=1) ** -0.5)
        A = torch.matmul(torch.matmul(D, A), D)
        self.register_buffer('A', A) 
        
        self.data_bn = nn.BatchNorm1d(in_channels * 17)
        self.layer1 = ST_GCN_Block(in_channels, 64)
        self.layer2 = ST_GCN_Block(64, 128, stride=2)
        self.layer3 = ST_GCN_Block(128, 256, stride=2)
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        x = self.layer1(x, self.A)
        x = self.layer2(x, self.A)
        x = self.layer3(x, self.A)
        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = self.fcn(x)
        return x.view(N, -1)


# --- INFERENCE SERVICE ---
class PoseInferenceService:
    def __init__(self, weights_filename: str = "stgcn_v1_production.ckpt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TT2SkeletonClassifier(num_classes=2).to(self.device)
        
        # 1. Check for an Environment Variable first (Production style)
        env_path = os.getenv("POSE_MODEL_PATH")
        
        if env_path:
            weights_path = Path(env_path)
        else:
            # 2. Fallback: Find weights relative to THIS file's location
            # This looks for the 'weights' folder in the same directory as analysis_stgcn.py
            weights_path = Path(__file__).resolve().parent / "weights" / weights_filename

        if not weights_path.exists():
            # Providing the absolute path in the error helps you debug instantly
            raise FileNotFoundError(f"ST-GCN weights not found at: {weights_path.absolute()}")

        # 3. Load the checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info(f"[ST-GCN] Loaded pose brain from: {weights_path.name}")

    def predict_sequence(self, sequence_15_frames: list) -> tuple:
        """
        Takes a list of 15 numpy arrays of shape (17, 2).
        Returns: (Action String, Confidence Float)
        """
        # 1. Stack and Format: (15, 17, 2)
        seq_np = np.array(sequence_15_frames) 
        
        # 2. Add Batch & Channel dims: (Batch=1, Channels=2, Frames=15, Joints=17)
        tensor_data = torch.from_numpy(seq_np).float().to(self.device)
        tensor_data = tensor_data.permute(2, 0, 1).unsqueeze(0) 

        # 3. Inference
        with torch.no_grad():
            logits = self.model(tensor_data)
            probs = torch.softmax(logits, dim=1)[0]
            
            threat_prob = probs[1].item()
            neutral_prob = probs[0].item()

        if threat_prob > 0.70: # Strict threshold for physical security
            return "punch", threat_prob
        return "neutral", neutral_prob