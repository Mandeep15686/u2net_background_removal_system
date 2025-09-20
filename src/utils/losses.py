
"""
Loss Functions for U²-Net Training
Custom loss implementations for multi-stage supervision and challenging scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class MultiStageBCELoss(nn.Module):
    """
    Multi-stage Binary Cross Entropy Loss for U²-Net deep supervision
    Computes BCE loss for all decoder outputs with equal weighting
    """

    def __init__(self):
        super(MultiStageBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, outputs: Tuple[torch.Tensor, ...], targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs: Tuple of (d0, d1, d2, d3, d4, d5, d6) from U²-Net
            targets: Ground truth masks
        """
        total_loss = 0.0

        for output in outputs:
            # Resize target to match output size if needed
            if output.size() != targets.size():
                target_resized = F.interpolate(targets, size=output.size()[2:], 
                                             mode='bilinear', align_corners=False)
            else:
                target_resized = targets

            loss = self.bce_loss(output, target_resized)
            total_loss += loss

        return total_loss

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation
    Focuses learning on hard examples
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IoULoss(nn.Module):
    """
    Intersection over Union Loss
    Directly optimizes IoU metric
    """

    def __init__(self, smooth: float = 1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum() - intersection

        # IoU calculation
        iou = (intersection + self.smooth) / (union + self.smooth)

        # Return 1 - IoU as loss (lower is better)
        return 1 - iou

class DiceLoss(nn.Module):
    """
    Dice Loss (F1 Score Loss)
    Good for segmentation tasks with class imbalance
    """

    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate Dice coefficient
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        # Return 1 - Dice as loss
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combined loss function using BCE + IoU + Dice
    Provides balanced optimization across different metrics
    """

    def __init__(self, bce_weight: float = 1.0, iou_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.dice_weight = dice_weight

        self.bce_loss = nn.BCELoss()
        self.iou_loss = IoULoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(inputs, targets)
        iou = self.iou_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)

        combined = (self.bce_weight * bce + 
                   self.iou_weight * iou + 
                   self.dice_weight * dice)

        return combined

class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss for preserving fine details
    Gives higher weight to boundary regions
    """

    def __init__(self, edge_weight: float = 2.0):
        super(EdgeAwareLoss, self).__init__()
        self.edge_weight = edge_weight
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute BCE loss
        bce = self.bce_loss(inputs, targets)

        # Detect edges in ground truth
        edges = self._detect_edges(targets)

        # Apply edge weighting
        weighted_loss = bce * (1 + self.edge_weight * edges)

        return weighted_loss.mean()

    def _detect_edges(self, masks: torch.Tensor) -> torch.Tensor:
        """Detect edges using Sobel operator"""
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=masks.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=masks.device).unsqueeze(0).unsqueeze(0)

        # Apply Sobel filters
        grad_x = F.conv2d(masks, sobel_x, padding=1)
        grad_y = F.conv2d(masks, sobel_y, padding=1)

        # Compute gradient magnitude
        edges = torch.sqrt(grad_x**2 + grad_y**2)

        # Normalize to [0, 1]
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)

        return edges
