
"""
Evaluation Metrics for U²-Net Background Removal
Comprehensive metrics for assessing model performance
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple, List
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Intersection over Union (IoU)"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary + target_binary - pred_binary * target_binary)

    iou = intersection / (union + 1e-8)
    return iou.item()

def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice coefficient (F1 score)"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = torch.sum(pred_binary * target_binary)
    dice = (2 * intersection) / (torch.sum(pred_binary) + torch.sum(target_binary) + 1e-8)

    return dice.item()

def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute pixel-wise accuracy"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    correct = torch.sum(pred_binary == target_binary)
    total = torch.numel(target_binary)

    accuracy = correct / total
    return accuracy.item()

def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error"""
    mae = torch.mean(torch.abs(pred - target))
    return mae.item()

def compute_boundary_accuracy(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    """Compute boundary accuracy within tolerance pixels"""
    pred_edges = extract_edges(pred)
    target_edges = extract_edges(target)

    # Dilate target edges by tolerance
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance*2+1, tolerance*2+1))
    target_dilated = cv2.dilate(target_edges.astype(np.uint8), kernel)

    # Count correct boundary predictions
    correct_boundary = np.sum(pred_edges & target_dilated.astype(bool))
    total_boundary = np.sum(pred_edges)

    boundary_acc = correct_boundary / (total_boundary + 1e-8)
    return boundary_acc

def extract_edges(mask: np.ndarray) -> np.ndarray:
    """Extract edges from mask using morphological operations"""
    if len(mask.shape) > 2:
        mask = mask.squeeze()

    # Ensure binary mask
    binary_mask = (mask > 0.5).astype(np.uint8) * 255

    # Extract edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary_mask, kernel)
    edges = binary_mask - eroded

    return edges > 0

def compute_comprehensive_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute all evaluation metrics"""
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()

    metrics = {
        'iou': compute_iou(pred, target),
        'dice': compute_dice(pred, target),
        'pixel_accuracy': compute_pixel_accuracy(pred, target),
        'mae': compute_mae(pred, target),
        'boundary_accuracy': compute_boundary_accuracy(pred_np, target_np)
    }

    # Additional threshold-based metrics
    for threshold in [0.3, 0.5, 0.7]:
        metrics[f'iou_@{threshold}'] = compute_iou(pred, target, threshold)
        metrics[f'dice_@{threshold}'] = compute_dice(pred, target, threshold)

    return metrics

class QualityAssurance:
    """Quality assurance and evaluation utilities"""

    def __init__(self):
        self.metrics_history = []

    def evaluate_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate a batch of predictions"""
        batch_metrics = {
            'iou': [],
            'dice': [],
            'pixel_accuracy': [],
            'mae': [],
            'boundary_accuracy': []
        }

        batch_size = predictions.size(0)

        for i in range(batch_size):
            pred = predictions[i]
            target = targets[i]

            metrics = compute_comprehensive_metrics(pred, target)

            for key in batch_metrics:
                if key in metrics:
                    batch_metrics[key].append(metrics[key])

        # Compute averages
        avg_metrics = {}
        for key, values in batch_metrics.items():
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)

        return avg_metrics

    def compute_dataset_statistics(self, dataloader, model, device) -> Dict[str, float]:
        """Compute statistics over entire dataset"""
        model.eval()
        all_metrics = []

        with torch.no_grad():
            for images, masks in dataloader:
                images, masks = images.to(device), masks.to(device)

                # Get predictions
                outputs = model(images)
                predictions = outputs[0]  # Main output

                # Compute metrics for batch
                batch_metrics = self.evaluate_batch(predictions, masks)
                all_metrics.append(batch_metrics)

        # Aggregate all metrics
        final_metrics = {}
        metric_keys = all_metrics[0].keys()

        for key in metric_keys:
            values = [metrics[key] for metrics in all_metrics]
            final_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return final_metrics

    def generate_evaluation_report(self, metrics: Dict) -> str:
        """Generate formatted evaluation report"""
        report = "="*60 + "\n"
        report += "U²-Net Model Evaluation Report\n"
        report += "="*60 + "\n\n"

        report += "Primary Metrics:\n"
        report += "-"*30 + "\n"

        primary_metrics = ['avg_iou', 'avg_dice', 'avg_pixel_accuracy', 'avg_mae', 'avg_boundary_accuracy']

        for metric in primary_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, dict):
                    report += f"{metric.replace('avg_', '').upper():.<20} {value['mean']:.4f} ± {value['std']:.4f}\n"
                else:
                    report += f"{metric.replace('avg_', '').upper():.<20} {value:.4f}\n"

        report += "\nThreshold Analysis:\n"
        report += "-"*30 + "\n"

        thresholds = [0.3, 0.5, 0.7]
        for threshold in thresholds:
            iou_key = f'iou_@{threshold}'
            dice_key = f'dice_@{threshold}'

            if iou_key in metrics:
                report += f"IoU @ {threshold}:...................{metrics[iou_key]:.4f}\n"
            if dice_key in metrics:
                report += f"Dice @ {threshold}:..................{metrics[dice_key]:.4f}\n"

        return report

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Main function for computing evaluation metrics"""
    return compute_comprehensive_metrics(pred, target)
