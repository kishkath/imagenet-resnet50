import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models
from typing import Dict, Any

from config import Config

class ResNet50Module(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = models.resnet50(num_classes=config.num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = pl.metrics.Accuracy(task='multiclass', num_classes=config.num_classes)
        self.val_acc = pl.metrics.Accuracy(task='multiclass', num_classes=config.num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
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
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        } 