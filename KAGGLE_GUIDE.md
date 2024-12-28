# Running ResNet50 Training on Kaggle

This guide provides step-by-step instructions for running the ResNet50 ImageNet training on Kaggle notebooks.

## 1. Kaggle Setup Requirements

### 1.1. Enable GPU and Internet
1. Click on the "..." menu in your notebook
2. Select "Settings"
3. Enable:
   - Internet: ON
   - GPU: P100 (or better if available)
   - Save settings

### 1.2. Set Up Environment
Add this at the start of your notebook:
```python
!pip install pytorch-lightning wandb albumentations --quiet
```

## 2. Clone Repository and Setup

```python
# Clone the repository
!git clone https://your-repo-url.git
%cd your-repo

# Install requirements
!pip install -r requirements.txt
```

## 3. Dataset Access

Kaggle provides ImageNet through their datasets API. Add this to your notebook:

```python
# Import necessary libraries
from kaggle.api.kaggle_api_extended import KaggleApi
import os

# Authenticate (make sure you have your kaggle.json in place)
api = KaggleApi()
api.authenticate()

# Create data directory
!mkdir -p /kaggle/working/imagenet

# Download ImageNet from Kaggle Datasets
!kaggle datasets download -d pytorch/imagenet
!unzip -q imagenet.zip -d /kaggle/working/imagenet
```

## 4. Modify Configuration

Create a cell with this configuration:

```python
from config import Config

config = Config()
config.data_dir = '/kaggle/working/imagenet'
config.batch_size = 128  # Adjust based on GPU memory
config.epochs = 100
config.num_workers = 2  # Kaggle typically allows 2-4 workers
config.use_amp = True
config.distributed = False  # Kaggle provides single GPU
```

## 5. Training Launch

### 5.1. Basic Training
```python
from trainer import train_imagenet

# Start training
train_imagenet(config)
```

### 5.2. With WandB Integration
```python
# Setup WandB
import wandb
wandb.login()  # You'll need to add your API key

# Configure WandB
config.use_wandb = True
config.project_name = 'imagenet_resnet50_kaggle'

# Start training
train_imagenet(config)
```

## 6. Monitoring Training

### 6.1. Real-Time Progress
The training progress will be displayed in real-time with:
- Progress bars for each epoch
- Detailed metrics every 10 batches
- Summary statistics at epoch end

Example output:
```
===============================================================================
Starting training epoch 0
===============================================================================
Epoch 0: 100%|██████████| 5004/5004 [09:23<00:00, 8.89it/s]
2023-XX-XX HH:MM:SS - INFO - Training - Epoch: 0, Step: 10/5004, Loss: 6.9074, Acc: 0.0078
2023-XX-XX HH:MM:SS - INFO - Training - Epoch: 0, Step: 20/5004, Loss: 6.8821, Acc: 0.0156

Finished training epoch 0:
========================================
Training Loss: 6.7234
Training Accuracy: 0.0234
========================================

--------------------------------------------------------------------------------
Starting validation epoch 0
--------------------------------------------------------------------------------
Validation 0: 100%|██████████| 50/50 [00:23<00:00, 2.13it/s]
2023-XX-XX HH:MM:SS - INFO - Validation - Batch: 0/50, Loss: 6.8123, Acc: 0.0195

Finished validation epoch 0:
----------------------------------------
Validation Loss: 6.7891
Validation Accuracy: 0.0273
----------------------------------------
```

### 6.2. Monitoring Options

1. **View Live Progress**:
```python
# The progress will be displayed automatically in the notebook output
```

2. **Check GPU Usage**:
```python
# In a separate cell, run:
!while true; do nvidia-smi; sleep 2; clear; done
```

3. **Monitor Memory**:
```python
# In a separate cell, run:
!while true; do nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 1; sleep 2; done
```

4. **View Log File**:
```python
# In a separate cell, run:
!tail -f training.log
```

### 6.3. Training Metrics Visualization

Run this in a separate cell to visualize training progress in real-time:
```python
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

def plot_metrics():
    while True:
        # Read log file
        metrics = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        with open('training.log', 'r') as f:
            for line in f:
                if 'Finished training epoch' in line:
                    # Parse training metrics
                    pass
                elif 'Finished validation epoch' in line:
                    # Parse validation metrics
                    pass
        
        # Create plots
        clear_output(wait=True)
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(metrics['epoch'], metrics['train_loss'], label='Train')
        plt.plot(metrics['epoch'], metrics['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(metrics['epoch'], metrics['train_acc'], label='Train')
        plt.plot(metrics['epoch'], metrics['val_acc'], label='Val')
        plt.title('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        time.sleep(10)  # Update every 10 seconds

# Run in a separate cell:
plot_metrics()
```

## 7. Saving Results

### 7.1. Save Checkpoints
```python
# Zip checkpoints
!zip -r /kaggle/working/checkpoints.zip /kaggle/working/checkpoints

# Download will be automatic in Kaggle interface
```

### 7.2. Save Logs
```python
# Zip logs
!zip -r /kaggle/working/logs.zip /kaggle/working/logs
```

## 8. Best Practices for Kaggle

1. **Memory Management**:
   - Use smaller batch size (128 or 64)
   - Gradient checkpointing is enabled by default in the model
   - Use mixed precision training
   ```python
   # If you need to manually adjust batch size for OOM errors
   config.batch_size = 64  # or even 32 if needed
   ```

2. **Time Management**:
   - Kaggle notebooks have runtime limits (usually 12 hours for GPU)
   - Save checkpoints frequently
   - Use notebook checkpointing

3. **Storage Management**:
   - Clean up unnecessary files:
   ```python
   !rm -rf /kaggle/working/imagenet
   ```
   - Keep only essential checkpoints

4. **Code for Resuming Training**:
```python
# Load from checkpoint
from trainer import train_imagenet
config.resume_from_checkpoint = '/kaggle/working/checkpoints/last.ckpt'
train_imagenet(config)
```

## 9. Complete Notebook Example

```python
# 1. Setup
!pip install pytorch-lightning wandb albumentations --quiet
!git clone https://your-repo-url.git
%cd your-repo
!pip install -r requirements.txt

# 2. Configure
from config import Config
config = Config()
config.data_dir = '/kaggle/working/imagenet'
config.batch_size = 128
config.epochs = 100
config.num_workers = 2
config.use_amp = True
config.distributed = False

# 3. Setup WandB (optional)
import wandb
wandb.login()
config.use_wandb = True
config.project_name = 'imagenet_resnet50_kaggle'

# 4. Start Training
from trainer import train_imagenet
train_imagenet(config)

# 5. Save Results
!zip -r /kaggle/working/checkpoints.zip /kaggle/working/checkpoints
!zip -r /kaggle/working/logs.zip /kaggle/working/logs
```

## 10. Troubleshooting

1. **OOM Errors**:
   ```python
   # Reduce batch size
   config.batch_size //= 2
   
   # Verify gradient checkpointing is enabled
   print("Gradient Checkpointing:", 
         any(getattr(m, 'gradient_checkpointing', False) 
             for m in model.model.modules()))
   ```

2. **Slow Training**:
   ```python
   # Monitor GPU utilization
   !nvidia-smi -l 1
   
   # Check memory usage
   !nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   ```

3. **Runtime Disconnection**:
   - Save checkpoints every epoch (already configured)
   - Use WandB for metric tracking
   - Keep notebook output minimal
   - Consider reducing validation frequency:
   ```python
   config.val_check_interval = 0.5  # Validate every 0.5 epochs
   ```

4. **Dataset Loading Issues**:
   ```python
   # Verify dataset is properly loaded
   print("Training samples:", len(train_loader.dataset))
   print("Validation samples:", len(val_loader.dataset))
   
   # Try reducing number of workers
   config.num_workers = 1
   ```

5. **Log Analysis**:
   ```python
   # Check for errors in log
   !grep "ERROR" training.log
   
   # View training progress
   !grep "Finished training epoch" training.log | tail -n 5
   
   # View validation progress
   !grep "Finished validation epoch" training.log | tail -n 5
   
   # Check learning rate changes
   !grep "learning_rate" training.log
   ```

6. **Training Progress Visualization**:
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   
   # Extract metrics from log
   def parse_metrics(log_file):
       train_metrics = []
       val_metrics = []
       with open(log_file, 'r') as f:
           for line in f:
               if 'Finished training epoch' in line:
                   # Parse and append training metrics
                   pass
               elif 'Finished validation epoch' in line:
                   # Parse and append validation metrics
                   pass
       return pd.DataFrame(train_metrics), pd.DataFrame(val_metrics)
   
   train_df, val_df = parse_metrics('training.log')
   
   # Plot metrics
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 2, 1)
   plt.plot(train_df['epoch'], train_df['loss'], label='Train')
   plt.plot(val_df['epoch'], val_df['loss'], label='Val')
   plt.title('Loss')
   plt.legend()
   
   plt.subplot(1, 2, 2)
   plt.plot(train_df['epoch'], train_df['acc'], label='Train')
   plt.plot(val_df['epoch'], val_df['acc'], label='Val')
   plt.title('Accuracy')
   plt.legend()
   plt.show()
   ``` 