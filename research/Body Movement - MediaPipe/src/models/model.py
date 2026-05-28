import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ST_GCN_Block(nn.Module):
    """
    [ARCHITECTURE] SOTA Spatial-Temporal Graph Convolutional Block.
    Performs spatial graph convolution followed by a temporal convolution.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1. Spatial Graph Convolution (Learning the body shape)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        
        # 2. Temporal Convolution (Learning the motion speed)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.1, inplace=True)
        )
        
        # Residual link to prevent vanishing gradients
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # x shape: (N, C, T, V) where V=17 joints
        res = self.residual(x)
        # Apply adjacency matrix (The body's connection map)
        x = self.gcn(x)
        x = torch.matmul(x, A) 
        x = self.tcn(x) + res
        return self.relu(x)

class TT2SkeletonClassifier(nn.Module):
    # UPGRADE: 4 Channels (X, Y, Z, Visibility)
    def __init__(self, num_classes=2, in_channels=4): 
        super().__init__()
        
        # UPGRADE: MediaPipe 33-Joint Adjacency Matrix
        A = torch.zeros((33, 33))
        mediapipe_pairs = [
            (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), 
            (11,12), (11,13), (13,15), (15,17), (15,19), (15,21), (17,19),
            (12,14), (14,16), (16,18), (16,20), (16,22), (18,20), (11,23), 
            (12,24), (23,24), (23,25), (24,26), (25,27), (26,28), (27,29),
            (28,30), (29,31), (30,32), (27,31), (28,32)
        ]
        for i in range(33): A[i, i] = 1.0
        for i, j in mediapipe_pairs:
            A[i, j] = 1.0; A[j, i] = 1.0
            
        D = torch.diag(torch.sum(A, dim=1) ** -0.5)
        A = torch.matmul(torch.matmul(D, A), D)
        self.register_buffer('A', A)
        
        # UPGRADE: 33 Joints
        self.data_bn = nn.BatchNorm1d(in_channels * 33) 

        self.layer1 = ST_GCN_Block(in_channels, 64)
        self.layer2 = ST_GCN_Block(64, 128, stride=2)
        self.layer3 = ST_GCN_Block(128, 256, stride=2)
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Input x: (N, C, T, V) -> (Batch, Coords, Frames, Joints)
        N, C, T, V = x.size()
        
        # 1. Coordinate Normalization
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # 2. Graph Convolution Layers
        x = self.layer1(x, self.A)
        x = self.layer2(x, self.A)
        x = self.layer3(x, self.A)

        # 3. Global Pooling and Logits
        x = x.mean(dim=(2, 3), keepdim=True) 
        x = self.fcn(x)
        return x.flatten(1)