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
    """
    [SOTA] The Brain of the Pose System.
    Uses Graph Convolutions to detect threats from 17 joint coordinates.
    """
    def __init__(self, num_classes=2, in_channels=2): # 2 Channels for (X, Y)
        super().__init__()
        
        # The COCO Adjacency Matrix (The Bones)
        # This maps how the 17 joints are physically connected
        A = torch.zeros((17, 17))
        coco_pairs = [
            (0,1), (0,2), (1,3), (2,4), (0,5), (0,6), (5,7), 
            (7,9), (6,8), (8,10), (5,11), (6,12), (11,13), 
            (13,15), (12,14), (14,16), (11,12)
        ]
        # Add self-loops (Identity)
        for i in range(17): A[i, i] = 1.0
        # Add bidirectional edges
        for i, j in coco_pairs:
            A[i, j] = 1.0
            A[j, i] = 1.0
            
        # Normalize the matrix
        D = torch.diag(torch.sum(A, dim=1) ** -0.5)
        A = torch.matmul(torch.matmul(D, A), D)
        self.register_buffer('A', A) 
        
        # Data Normalization Layer: Moves pelvis to (0,0) mathematically
        self.data_bn = nn.BatchNorm1d(in_channels * 17)

        # Multi-layer Graph Processing
        self.layer1 = ST_GCN_Block(in_channels, 64)
        self.layer2 = ST_GCN_Block(64, 128, stride=2)
        self.layer3 = ST_GCN_Block(128, 256, stride=2)

        # Final Classification Head
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
        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = self.fcn(x)
        return x.view(N, -1)