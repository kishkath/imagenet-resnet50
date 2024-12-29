import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models
import sys
from tqdm import tqdm
import torchmetrics
from utils import safe_cuda_memory_check, TrainingError, setup_logging
import logging

class Config:
    def __init__(self, num_classes, learning_rate, momentum, weight_decay, max_lr, epochs, pct_start, div_factor, final_div_factor, three_phase=True, anneal_strategy='cos'):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        self.epochs = epochs
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.anneal_strategy = anneal_strategy

class ResNet50Module(pl.LightningModule):
    def __init__(self, config: Config):
        try:
            super().__init__()
            self.config = config
            self.save_hyperparameters()
            
            # Setup custom logger
            self._custom_logger = setup_logging()
            self._custom_logger.info("Initializing ResNet50 model...")
            
            # Create model
            self.model = models.resnet50(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, config.num_classes)
            self._custom_logger.info(f"Model created with {config.num_classes} output classes")
            
            # Loss function
            self.criterion = nn.CrossEntropyLoss()
            
            # Initialize metrics
            self._init_metrics()
            
            self._custom_logger.info("Model initialization completed")
            
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
            
            self._custom_logger.info("Metrics initialized successfully")
            
        except Exception as e:
            self._custom_logger.error(f"Error initializing metrics: {str(e)}")
            raise
        
    def forward(self, x):
        try:
            return self.model(x)
        except Exception as e:
            self._custom_logger.error(f"Error in forward pass: {str(e)}")
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
            
            # Log metrics
            self.log('loss', loss, prog_bar=True)
            self.log('acc', acc, prog_bar=True)
            
            # Log to file periodically
            if batch_idx % 100 == 0:
                self._custom_logger.info(
                    f"Training - Epoch: {self.current_epoch}, "
                    f"Batch: {batch_idx}, Loss: {loss:.4f}, Acc: {acc:.4f}"
                )
            
            return {'loss': loss, 'acc': acc}
            
        except Exception as e:
            self._custom_logger.error(f"Error in training step: {str(e)}")
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
            
            # Log metrics
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)
            
            # Log to file periodically
            if batch_idx % 50 == 0:
                self._custom_logger.info(
                    f"Validation - Epoch: {self.current_epoch}, "
                    f"Batch: {batch_idx}, Loss: {loss:.4f}, Acc: {acc:.4f}"
                )
            
            return {'loss': loss, 'acc': acc}
            
        except Exception as e:
            self._custom_logger.error(f"Error in validation step: {str(e)}")
            raise
    
    def on_train_end(self):
        """Called when training ends"""
        try:
            self._custom_logger.info("\nTraining completed. Generating visualizations...")
            
            # Import here to avoid circular imports
            from visualize import visualize_training
            
            # Get the log file path from the logger
            log_file = None
            for handler in self._custom_logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    log_file = handler.baseFilename
                    break
            
            if log_file:
                visualize_training(log_file)
                self._custom_logger.info("Training visualizations generated successfully")
            else:
                self._custom_logger.warning("Could not find log file for visualization")
                
        except Exception as e:
            self._custom_logger.error(f"Error generating training visualizations: {str(e)}")
            
    def on_train_epoch_end(self):
        try:
            train_loss = self.train_loss.compute()
            train_acc = self.train_acc.compute()
            
            # Log epoch results
            self._custom_logger.info(f"\nTraining Epoch {self.current_epoch} Results:")
            self._custom_logger.info(f"Loss: {train_loss:.4f}")
            self._custom_logger.info(f"Accuracy: {train_acc:.4f}")
            
            # Log GPU memory stats
            allocated, cached = safe_cuda_memory_check()
            if allocated > 0:
                self._custom_logger.info(
                    f"GPU Memory: {allocated:.1f}GB allocated, "
                    f"{cached:.1f}GB cached"
                )
            
            self.train_loss.reset()
            self.train_acc.reset()
            
        except Exception as e:
            self._custom_logger.error(f"Error in train epoch end: {str(e)}")
    
    def on_validation_epoch_end(self):
        try:
            val_loss = self.val_loss.compute()
            val_acc = self.val_acc.compute()
            
            # Log epoch results
            self._custom_logger.info(f"\nValidation Epoch {self.current_epoch} Results:")
            self._custom_logger.info(f"Loss: {val_loss:.4f}")
            self._custom_logger.info(f"Accuracy: {val_acc:.4f}")
            
            self.val_loss.reset()
            self.val_acc.reset()
            
        except Exception as e:
            self._custom_logger.error(f"Error in validation epoch end: {str(e)}")
    
    def configure_optimizers(self):
        try:
            self._custom_logger.info("\nConfiguring optimizer and scheduler...")
            
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
            
            self._custom_logger.info(f"Training steps per epoch: {steps_per_epoch}")
            self._custom_logger.info(f"Total training steps: {total_steps}")
            
            # Log learning rate schedule parameters
            self._custom_logger.info("\nLearning Rate Schedule Parameters:")
            self._custom_logger.info(f"Initial learning rate: {self.config.max_lr/self.config.div_factor:.6f}")
            self._custom_logger.info(f"Maximum learning rate: {self.config.max_lr:.6f}")
            self._custom_logger.info(f"Final learning rate: {self.config.max_lr/(self.config.div_factor*self.config.final_div_factor):.6f}")
            self._custom_logger.info(f"Warmup epochs: {int(self.config.epochs * self.config.pct_start)}")
            self._custom_logger.info(f"Annealing epochs: {self.config.epochs - int(self.config.epochs * self.config.pct_start)}")
            self._custom_logger.info(f"Three phase: {self.config.three_phase}")
            self._custom_logger.info(f"Annealing strategy: {self.config.anneal_strategy}")
            
            # Scheduler with three-phase learning rate
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.max_lr,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.config.pct_start,
                div_factor=self.config.div_factor,
                final_div_factor=self.config.final_div_factor,
                three_phase=self.config.three_phase,
                anneal_strategy=self.config.anneal_strategy
            )
            
            self._custom_logger.info("Optimizer and scheduler configuration completed")
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
            
        except Exception as e:
            self._custom_logger.error(f"Error configuring optimizers: {str(e)}")
            raise 