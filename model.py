import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models
from typing import Dict, Any
import torchmetrics
import logging
import sys
from tqdm import tqdm

# Setup logging with real-time display
class RealTimeFormatter(logging.Formatter):
    def format(self, record):
        # Force flush after each log message
        formatted_message = super().format(record)
        sys.stdout.flush()
        return formatted_message

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler with custom formatter
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(RealTimeFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('training.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Progress bar for batch processing
class ProgressBarLogger:
    def __init__(self, total, desc=""):
        self.pbar = tqdm(total=total, desc=desc, leave=True)
    
    def update(self, n=1):
        self.pbar.update(n)
        self.pbar.refresh()
    
    def close(self):
        self.pbar.close()

from config import Config

class ResNet50Module(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        logger.info("Initializing ResNet50 model...")
        
        # Create model
        self.model = models.resnet50(num_classes=config.num_classes)
        logger.info(f"Model created with {config.num_classes} output classes")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=config.num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=config.num_classes)
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        
        logger.info("Model initialization completed")
        
        # Progress tracking
        self.train_progress = None
        self.val_progress = None
    
    def on_train_epoch_start(self):
        logger.info(f"\n{'='*80}\nStarting training epoch {self.current_epoch}\n{'='*80}")
        total_batches = len(self.trainer.train_dataloader)
        self.train_progress = ProgressBarLogger(total_batches, f"Epoch {self.current_epoch}")
    
    def on_validation_epoch_start(self):
        logger.info(f"\n{'-'*80}\nStarting validation epoch {self.current_epoch}\n{'-'*80}")
        total_batches = len(self.trainer.val_dataloaders[0])
        self.val_progress = ProgressBarLogger(total_batches, f"Validation {self.current_epoch}")
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)
        self.train_loss(loss)
        
        # Log metrics
        self.log('train_loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update progress bar
        if self.train_progress:
            self.train_progress.update()
        
        # Detailed logging every n steps
        if batch_idx % 10 == 0:  # Increased frequency for better visibility
            logger.info(f"Training - Epoch: {self.current_epoch}, Step: {batch_idx}/{len(self.trainer.train_dataloader)}, "
                       f"Loss: {loss:.4f}, Acc: {acc:.4f}")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        self.val_loss(loss)
        
        # Log metrics
        self.log('val_loss', self.val_loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        
        # Update progress bar
        if self.val_progress:
            self.val_progress.update()
        
        # Log first batch and every 10th batch
        if batch_idx % 10 == 0:
            logger.info(f"Validation - Batch: {batch_idx}/{len(self.trainer.val_dataloaders[0])}, "
                       f"Loss: {loss:.4f}, Acc: {acc:.4f}")
        
        return loss
    
    def on_train_epoch_end(self):
        if self.train_progress:
            self.train_progress.close()
        
        train_loss = self.train_loss.compute()
        train_acc = self.train_acc.compute()
        
        logger.info(f"\nFinished training epoch {self.current_epoch}:")
        logger.info(f"{'='*40}")
        logger.info(f"Training Loss: {train_loss:.4f}")
        logger.info(f"Training Accuracy: {train_acc:.4f}")
        logger.info(f"{'='*40}\n")
        
        self.train_loss.reset()
        self.train_acc.reset()
    
    def on_validation_epoch_end(self):
        if self.val_progress:
            self.val_progress.close()
        
        val_loss = self.val_loss.compute()
        val_acc = self.val_acc.compute()
        
        logger.info(f"\nFinished validation epoch {self.current_epoch}:")
        logger.info(f"{'-'*40}")
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_acc:.4f}")
        logger.info(f"{'-'*40}\n")
        
        self.val_loss.reset()
        self.val_acc.reset()
    
    def configure_optimizers(self):
        logger.info("Configuring optimizer and scheduler...")
        
        # Optimizer
        optimizer = SGD(
            self.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Calculate total steps for OneCycleLR
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        total_steps = self.trainer.estimated_stepping_batches
        
        logger.info(f"Training will run for {total_steps} total steps, "
                   f"{steps_per_epoch} steps per epoch")
        
        # Scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.max_lr,
            epochs=self.config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.config.pct_start,
            div_factor=self.config.div_factor,
            final_div_factor=self.config.final_div_factor
        )
        
        logger.info("Optimizer and scheduler configuration completed")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        } 