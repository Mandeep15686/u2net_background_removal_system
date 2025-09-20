
"""
U²-Net Training Script
Team 1: The Isolationists - Background Removal Training Pipeline

This script provides comprehensive training functionality for U²-Net model
with multi-stage supervision, edge case handling, and performance optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
import os
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import wandb

# Local imports
from ..models.u2net_model import U2NET, SaliencyDataset
from ..models.edge_case_handlers import AdvancedDataAugmentation, EdgeCaseHandler
from ..utils.losses import MultiStageBCELoss, IoULoss, FocalLoss
from ..utils.metrics import compute_metrics, QualityAssurance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class U2NetTrainer:
    """Comprehensive training class for U²-Net model"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize model
        self.model = U2NET(
            in_ch=config['model']['input_channels'],
            out_ch=config['model']['output_channels']
        ).to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Initialize losses
        self.criterion = self._create_loss_function()

        # Initialize metrics tracking
        self.best_iou = 0.0
        self.training_history = {'train_loss': [], 'val_loss': [], 'val_iou': []}

        # Initialize tensorboard writer
        self.writer = SummaryWriter(config['logging']['tensorboard_dir'])

        # Initialize quality assurance
        self.qa = QualityAssurance()

        self.logger.info(f"Initialized U2NetTrainer on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        optimizer_name = self.config['training']['optimizer']['name'].lower()
        lr = self.config['training']['optimizer']['learning_rate']
        weight_decay = self.config['training']['optimizer'].get('weight_decay', 1e-4)

        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config['training']['optimizer'].get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        scheduler_config = self.config['training'].get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'steplr').lower()

        if scheduler_name == 'steplr':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = scheduler_config.get('T_max', 50)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_name == 'plateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor
            )
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=1.0)  # No scheduling

    def _create_loss_function(self) -> nn.Module:
        """Create loss function with multi-stage supervision"""
        loss_config = self.config['training']['loss']
        loss_type = loss_config.get('type', 'bce').lower()

        if loss_type == 'bce':
            return MultiStageBCELoss()
        elif loss_type == 'focal':
            alpha = loss_config.get('alpha', 1.0)
            gamma = loss_config.get('gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'iou':
            return IoULoss()
        elif loss_type == 'combined':
            return CombinedLoss()
        else:
            return MultiStageBCELoss()

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        data_config = self.config['data']

        # Data augmentation
        augmentation = AdvancedDataAugmentation(image_size=data_config['image_size'])

        # Training transforms
        train_transform = transforms.Compose([
            transforms.Resize((data_config['image_size'], data_config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Validation transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((data_config['image_size'], data_config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        full_dataset = SaliencyDataset(
            image_dir=data_config['image_dir'],
            mask_dir=data_config['mask_dir'],
            transform=None,  # We'll handle transforms separately
            image_size=data_config['image_size']
        )

        # Split dataset
        train_size = int(data_config['train_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )

        self.logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(self.device), masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Compute multi-stage loss
            loss = self.criterion(outputs, masks)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config['training'].get('gradient_clipping', False):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip_value']
                )

            self.optimizer.step()

            total_loss += loss.item()

            # Log batch progress
            if batch_idx % self.config['logging']['log_interval'] == 0:
                progress = 100.0 * batch_idx / num_batches
                self.logger.info(
                    f'Epoch {epoch}: [{batch_idx}/{num_batches} '
                    f'({progress:.1f}%)]\tLoss: {loss.item():.6f}'
                )

                # Log to tensorboard
                step = epoch * num_batches + batch_idx
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), step)

        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)

        return avg_loss

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Compute loss
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                # Compute IoU for main output
                main_output = outputs[0]
                iou = self.qa.compute_iou(main_output, masks)
                total_iou += iou

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        # Log to tensorboard
        self.writer.add_scalar('Loss/Val_Epoch', avg_loss, epoch)
        self.writer.add_scalar('IoU/Val_Epoch', avg_iou, epoch)

        return avg_loss, avg_iou

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        save_dir = Path(self.config['model']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Training
            train_loss = self.train_epoch(train_loader, epoch)

            # Validation
            val_loss, val_iou = self.validate_epoch(val_loader, epoch)

            # Update learning rate scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_iou'].append(val_iou)

            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            self.logger.info(
                f'Epoch {epoch+1}/{num_epochs} - '
                f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                f'Val IoU: {val_iou:.6f}, LR: {current_lr:.2e}, '
                f'Time: {epoch_time:.2f}s'
            )

            # Save best model
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                best_model_path = save_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_iou': self.best_iou,
                    'config': self.config
                }, best_model_path)

                self.logger.info(f'New best model saved with IoU: {val_iou:.6f}')

            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'training_history': self.training_history,
                    'config': self.config
                }, checkpoint_path)

        # Save final model
        final_model_path = save_dir / 'final_model.pth'
        torch.save(self.model.state_dict(), final_model_path)

        # Save training history
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        self.writer.close()

        return {
            'best_iou': self.best_iou,
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'training_history': self.training_history
        }

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'best_iou' in checkpoint:
            self.best_iou = checkpoint['best_iou']

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']

def create_default_config() -> Dict:
    """Create default training configuration"""
    return {
        'model': {
            'input_channels': 3,
            'output_channels': 1,
            'save_dir': 'checkpoints'
        },
        'data': {
            'image_dir': 'data/images',
            'mask_dir': 'data/masks',
            'image_size': 320,
            'batch_size': 8,
            'train_split': 0.8,
            'num_workers': 4
        },
        'training': {
            'num_epochs': 100,
            'save_interval': 10,
            'gradient_clipping': True,
            'gradient_clip_value': 1.0,
            'optimizer': {
                'name': 'Adam',
                'learning_rate': 0.001,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'name': 'StepLR',
                'step_size': 30,
                'gamma': 0.1
            },
            'loss': {
                'type': 'bce'
            }
        },
        'logging': {
            'log_interval': 10,
            'tensorboard_dir': 'runs/u2net_training'
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Train U²-Net for Background Removal')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--image-dir', type=str, help='Directory containing training images')
    parser.add_argument('--mask-dir', type=str, help='Directory containing training masks')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save models')

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()

        # Override with command line arguments
        if args.image_dir:
            config['data']['image_dir'] = args.image_dir
        if args.mask_dir:
            config['data']['mask_dir'] = args.mask_dir
        if args.epochs:
            config['training']['num_epochs'] = args.epochs
        if args.batch_size:
            config['data']['batch_size'] = args.batch_size
        if args.lr:
            config['training']['optimizer']['learning_rate'] = args.lr
        if args.save_dir:
            config['model']['save_dir'] = args.save_dir

    print("Training Configuration:")
    print(json.dumps(config, indent=2))

    # Initialize trainer
    trainer = U2NetTrainer(config)

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # Create data loaders
    train_loader, val_loader = trainer.create_dataloaders()

    # Start training
    print(f"\nStarting training from epoch {start_epoch}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Device: {trainer.device}")

    results = trainer.train(train_loader, val_loader)

    print(f"\nTraining completed!")
    print(f"Best validation IoU: {results['best_iou']:.6f}")
    print(f"Final training loss: {results['final_train_loss']:.6f}")
    print(f"Final validation loss: {results['final_val_loss']:.6f}")

if __name__ == "__main__":
    main()
