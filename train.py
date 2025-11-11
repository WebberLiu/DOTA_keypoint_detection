"""
Training script for DOTA keypoint detection model.

This script handles:
- Model initialization
- Training loop with logging
- Validation during training
- Checkpointing
- TensorBoard logging

Usage:
    python train.py
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import (
    DATA_CONFIG, TRAIN_CONFIG, MODEL_CONFIG, LOGGING_CONFIG,
    DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR
)
from data import create_dataloaders, get_train_transforms, get_val_transforms
from models import create_model
from utils.metrics import compute_metrics
from utils.visualization import plot_training_curves


class Trainer:
    """
    Trainer class for keypoint detection model.
    
    Handles the complete training pipeline including:
    - Model training and validation
    - Metric tracking
    - Checkpointing
    - TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        device: str,
        log_dir: str
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Logging
        self.writer = SummaryWriter(log_dir) if LOGGING_CONFIG["tensorboard"] else None
        
        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {
            "mae": [],
            "mse": [],
            "pck@10": [],
            "pck@20": [],
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        
        for batch_idx, (images, heatmaps, metadata) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            heatmaps = heatmaps.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_heatmaps = self.model(images)
            loss = self.criterion(pred_heatmaps, heatmaps)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to TensorBoard
            if self.writer and batch_idx % TRAIN_CONFIG["log_every"] == 0:
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def validate(self, val_loader, epoch: int) -> tuple:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        # Collect predictions for metrics
        all_pred_keypoints = []
        all_gt_keypoints = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")
        
        with torch.no_grad():
            for images, heatmaps, metadata in pbar:
                # Move to device
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                
                # Forward pass
                pred_heatmaps = self.model(images)
                loss = self.criterion(pred_heatmaps, heatmaps)
                
                total_loss += loss.item()
                
                # Extract keypoints from predicted heatmaps
                pred_keypoints = self.model.extract_keypoints(
                    pred_heatmaps,
                    threshold=0.3,
                    nms_threshold=10
                )
                
                # Collect for metrics
                all_pred_keypoints.extend(pred_keypoints)
                all_gt_keypoints.extend([np.array(m["keypoints"]) for m in metadata])
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        # Compute metrics
        metrics = compute_metrics(
            all_pred_keypoints,
            all_gt_keypoints,
            image_size=DATA_CONFIG["image_size"],
            heatmap_size=MODEL_CONFIG["heatmap_size"]
        )
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        checkpoint_dir: str
    ):
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            
            # Track metrics
            for key in self.metrics_history.keys():
                if key in metrics:
                    self.metrics_history[key].append(metrics[key])
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("train/epoch_loss", train_loss, epoch)
                self.writer.add_scalar("val/epoch_loss", val_loss, epoch)
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"val/{key}", value, epoch)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("train/learning_rate", current_lr, epoch)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  MAE:        {metrics.get('mae', 0):.2f} pixels")
            print(f"  PCK@10:     {metrics.get('pck@10', 0):.4f}")
            print(f"  PCK@20:     {metrics.get('pck@20', 0):.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            
            if (epoch + 1) % TRAIN_CONFIG["save_every"] == 0 or is_best:
                self.save_checkpoint(
                    epoch,
                    checkpoint_dir,
                    is_best=is_best,
                    metrics=metrics
                )
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time / 60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch + 1})")
        print(f"{'='*70}\n")
        
        # Save training curves
        self.save_training_curves(checkpoint_dir)
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(
        self,
        epoch: int,
        checkpoint_dir: str,
        is_best: bool = False,
        metrics: dict = None
    ):
        """Save model checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "metrics_history": self.metrics_history,
            "best_val_loss": self.best_val_loss,
            "config": {
                "model": MODEL_CONFIG,
                "train": TRAIN_CONFIG,
                "data": DATA_CONFIG,
            }
        }
        
        if metrics:
            checkpoint["latest_metrics"] = metrics
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  Best model saved: {best_path}")
    
    def save_training_curves(self, save_dir: str):
        """Save training curves as image."""
        save_dir = Path(save_dir)
        save_path = save_dir / "training_curves.png"
        
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.metrics_history,
            save_path=str(save_path)
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DOTA keypoint detection model")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_epochs"],
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"],
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"],
                       help="Learning rate")
    parser.add_argument("--device", type=str, default=TRAIN_CONFIG["device"],
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Set device
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=str(DATA_DIR),
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms(),
        generate_synthetic=False  # Will auto-fallback to synthetic if DOTA not found
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model and loss
    print("\nInitializing model...")
    model, criterion = create_model(device=device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAIN_CONFIG["lr_step_size"],
        gamma=TRAIN_CONFIG["lr_gamma"]
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    
    # Create trainer
    log_dir = Path(LOGS_DIR) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=str(log_dir)
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        checkpoint_dir=str(CHECKPOINTS_DIR)
    )
    
    print("\nTraining script completed successfully!")
    print(f"Checkpoints saved to: {CHECKPOINTS_DIR}")
    print(f"Logs saved to: {log_dir}")
    print(f"\nTo view training logs, run:")
    print(f"  tensorboard --logdir {LOGS_DIR}")


if __name__ == "__main__":
    main()

