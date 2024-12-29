import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models
import sys
from tqdm import tqdm
import torchmetrics
from utils import safe_cuda_memory_check, TrainingError

class Config:
    def __init__(self, num_classes, learning_rate, momentum, weight_decay, max_lr, epochs, pct_start, div_factor, final_div_factor):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        self.epochs = epochs
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

class ResNet50Module(pl.LightningModule):
    def __init__(self, config: Config):
        try:
            super().__init__()
            self.config = config
            self.save_hyperparameters()
            
            print("Initializing ResNet50 model...")
            
            # Create model
            # self.model = models.resnet50(num_classes=config.num_classes)
            self.model = models.resnet50(weights=None)
            model.fc = nn.Linear(self.model.fc.in_features, config.num_classes)
            print(f"Model created with {config.num_classes} output classes")
            
            # Loss function
            self.criterion = nn.CrossEntropyLoss()
            
            # Initialize metrics
            self._init_metrics()
            
            print("Model initialization completed")
            
        except Exception as e:
            print(f"Error initializing ResNet50Module: {str(e)}")
            raise
            
    def _init_metrics(self):
        try:
            # Metrics
            self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.config.num_classes)
            self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.config.num_classes)
            self.train_loss = torchmetrics.MeanMetric()
            self.val_loss = torchmetrics.MeanMetric()
            
            # Store current metrics
            self.current_train_loss = 0.0
            self.current_train_acc = 0.0
            self.current_val_loss = 0.0
            self.current_val_acc = 0.0
        except Exception as e:
            print(f"Error initializing metrics: {str(e)}")
            raise
        
    def forward(self, x):
        try:
            return self.model(x)
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise
    
    def training_step(self, batch, batch_idx):
        try:
            images, labels = batch
            
            # Forward pass
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            
            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = self.train_acc(preds, labels)
            self.train_loss(loss)
            
            # Store current metrics
            self.current_train_loss = loss.item()
            self.current_train_acc = acc.item()
            
            # Log to progress bar
            self.log('loss', loss, prog_bar=True)
            self.log('acc', acc, prog_bar=True)
            
            return {'loss': loss, 'acc': acc}
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            raise
    
    def validation_step(self, batch, batch_idx):
        try:
            images, labels = batch
            
            # Forward pass
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            
            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = self.val_acc(preds, labels)
            self.val_loss(loss)
            
            # Store current metrics
            self.current_val_loss = loss.item()
            self.current_val_acc = acc.item()
            
            # Log to progress bar
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)
            
            return {'loss': loss, 'acc': acc}
            
        except Exception as e:
            print(f"Error in validation step: {str(e)}")
            raise
    
    def on_train_epoch_end(self):
        try:
            train_loss = self.train_loss.compute()
            train_acc = self.train_acc.compute()
            
            print(f"\nTraining Epoch {self.current_epoch} Results:")
            print(f"Loss: {train_loss:.4f}")
            print(f"Accuracy: {train_acc:.4f}")
            
            # Print GPU memory stats
            allocated, cached = safe_cuda_memory_check()
            if allocated > 0:
                print(f"GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            
            self.train_loss.reset()
            self.train_acc.reset()
            
        except Exception as e:
            print(f"Error in train epoch end: {str(e)}")
    
    def on_validation_epoch_end(self):
        try:
            val_loss = self.val_loss.compute()
            val_acc = self.val_acc.compute()
            
            print(f"\nValidation Epoch {self.current_epoch} Results:")
            print(f"Loss: {val_loss:.4f}")
            print(f"Accuracy: {val_acc:.4f}")
            
            self.val_loss.reset()
            self.val_acc.reset()
            
        except Exception as e:
            print(f"Error in validation epoch end: {str(e)}")
    
    def configure_optimizers(self):
        try:
            print("\nConfiguring optimizer and scheduler...")
            
            # Optimizer
            optimizer = SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
            
            # Calculate total steps for OneCycleLR
            if not hasattr(self.trainer, 'estimated_stepping_batches'):
                raise TrainingError("Trainer not properly initialized")
                
            steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            total_steps = self.trainer.estimated_stepping_batches
            
            print(f"Training steps per epoch: {steps_per_epoch}")
            print(f"Total training steps: {total_steps}")
            
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
            
            print("Optimizer and scheduler configuration completed")
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
            
        except Exception as e:
            print(f"Error configuring optimizers: {str(e)}")
            raise 
