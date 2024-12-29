import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import sys
from datetime import datetime
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Import local modules
from config import Config
from dataset import ImageNetDataset
from model import ResNet50Module
from utils import (
    ensure_directory, check_gpu_availability, safe_cuda_memory_check,
    cleanup_checkpoints, save_backup, check_system_resources,
    TrainingError, DatasetError
)

class ConsoleLogger:
    def __init__(self):
        try:
            self.start_time = None
            self.epoch_start_time = None
            self.last_log_time = None
            self.train_losses = []
            self.val_losses = []
            self.train_accs = []
            self.val_accs = []
        except Exception as e:
            print(f"Error initializing ConsoleLogger: {str(e)}")
            raise
    
    def start_training(self):
        try:
            self.start_time = time.time()
            self._print_header("Starting Training")
            has_gpu, gpu_name, memory = check_gpu_availability()
            if has_gpu:
                print(f"Using GPU: {gpu_name} ({memory:.1f}GB)")
                print(f"CUDA Version: {torch.version.cuda}")
        except Exception as e:
            print(f"Error in start_training: {str(e)}")
    
    def _print_header(self, text, width=100):
        try:
            print(f"\n{'='*width}")
            print(f"{text:^{width}}")
            print(f"{'='*width}\n")
        except Exception as e:
            print(f"Error printing header: {str(e)}")
    
    def _format_time(self, seconds):
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        except Exception as e:
            print(f"Error formatting time: {str(e)}")
            return "00:00:00"
    
    def log_batch(self, epoch, batch_idx, total_batches, loss, acc, lr):
        try:
            current_time = time.time()
            
            if self.last_log_time is None or (current_time - self.last_log_time) >= 1:
                elapsed = current_time - self.start_time
                epoch_elapsed = current_time - self.epoch_start_time
                
                batches_done = batch_idx + 1
                speed = batches_done / epoch_elapsed if epoch_elapsed > 0 else 0
                remaining_batches = total_batches - batches_done
                epoch_remaining = remaining_batches / speed if speed > 0 else 0
                
                # Get GPU memory info
                allocated, cached = safe_cuda_memory_check()
                
                status = (
                    f"\rEpoch {epoch+1} | "
                    f"Batch {batch_idx+1}/{total_batches} | "
                    f"Loss: {loss:.4f} | "
                    f"Acc: {acc:.4f} | "
                    f"LR: {lr:.6f} | "
                    f"GPU Mem: {allocated:.1f}GB | "
                    f"Elapsed: {self._format_time(elapsed)} | "
                    f"Remaining: {self._format_time(epoch_remaining)}"
                )
                
                print(status, end="", flush=True)
                self.last_log_time = current_time
                
        except Exception as e:
            print(f"\nError logging batch: {str(e)}")

class ImageNetTrainer:
    def __init__(self, config: Config):
        try:
            self.config = config
            self.logger = ConsoleLogger()
            print("Initializing ImageNet Trainer...")
            
            # Ensure directories exist
            ensure_directory(self.config.save_dir)
            ensure_directory(Path(self.config.data_dir))
            
            # Check system resources
            if not check_system_resources():
                raise TrainingError("System resource check failed")
                
        except Exception as e:
            print(f"Error initializing trainer: {str(e)}")
            raise
        
    def setup(self):
        try:
            print("Setting up training...")
            
            # Initialize dataset
            dataset = ImageNetDataset(self.config)
            self.train_loader, self.val_loader = dataset.get_dataloaders(
                distributed=self.config.distributed
            )
            
            if not self.train_loader or not self.val_loader:
                raise DatasetError("Failed to initialize data loaders")
                
            print(f"Dataset loaded - Training samples: {len(self.train_loader.dataset)}, "
                  f"Validation samples: {len(self.val_loader.dataset)}")
            
            # Initialize model
            self.model = ResNet50Module(config=self.config)
            
            # Enable gradient checkpointing
            try:
                if hasattr(self.model.model, 'set_grad_checkpointing'):
                    self.model.model.set_grad_checkpointing(enable=True)
                    print("Gradient checkpointing enabled")
            except Exception as e:
                print(f"Warning: Could not enable gradient checkpointing: {str(e)}")
            
            # Setup callbacks with error handling
            self.setup_callbacks()
            
        except Exception as e:
            print(f"Error in setup: {str(e)}")
            raise
            
    def setup_callbacks(self):
        try:
            self.callbacks = [
                ModelCheckpoint(
                    dirpath=self.config.save_dir,
                    filename='resnet50-{epoch:02d}-{val_acc:.3f}',
                    monitor='val_acc',
                    mode='max',
                    save_top_k=3,
                    every_n_epochs=self.config.save_freq
                ),
                LearningRateMonitor(logging_interval='step')
            ]
        except Exception as e:
            print(f"Error setting up callbacks: {str(e)}")
            raise
        
    def train(self):
        try:
            self.logger.start_training()
            
            # Create backup of important files
            save_backup('config.py')
            save_backup('model.py')
            
            # Initialize trainer
            trainer = pl.Trainer(
                max_epochs=self.config.epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                precision=16 if self.config.use_amp else 32,
                callbacks=self.callbacks,
                enable_progress_bar=self.config.show_progress_bar,
                log_every_n_steps=self.config.log_every_n_steps,
                val_check_interval=self.config.val_check_interval
            )
            
            # Print training configuration
            self.print_training_config()
            
            # Train
            trainer.fit(
                model=self.model,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader
            )
            
            # Cleanup old checkpoints
            cleanup_checkpoints(self.config.save_dir)
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
            
    def print_training_config(self):
        try:
            print("\nTraining Configuration:")
            print(f"Batch Size: {self.config.batch_size}")
            print(f"Learning Rate: {self.config.learning_rate}")
            print(f"Mixed Precision: {self.config.use_amp}")
            print(f"Gradient Clipping: {self.config.gradient_clip_val}")
            print(f"Validation Check Interval: {self.config.val_check_interval}")
        except Exception as e:
            print(f"Error printing configuration: {str(e)}")

def train_imagenet(config: Config):
    try:
        print("\nStarting ImageNet training process...")
        
        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Create and setup trainer
        trainer = ImageNetTrainer(config)
        trainer.setup()
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"Fatal error in training process: {str(e)}")
        raise 