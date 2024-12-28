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

### 6.1. GPU Monitoring
```python
# Monitor GPU usage
!nvidia-smi
```

### 6.2. Training Progress
```python
# View latest training metrics
!tail -f /kaggle/working/training.log
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
   - Enable gradient checkpointing
   - Use mixed precision training

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
   config.batch_size //= 2  # Reduce batch size
   config.gradient_checkpointing = True
   ```

2. **Slow Training**:
   ```python
   # Monitor GPU utilization
   !nvidia-smi -l 1
   ```

3. **Runtime Disconnection**:
   - Save checkpoints every epoch
   - Use WandB for metric tracking
   - Keep notebook output minimal 