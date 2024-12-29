# Step-by-Step Guide: Training ResNet50 in Kaggle Notebook

## Step 1: Set Up Kaggle Environment

1. Create a new notebook in Kaggle:
   - Go to Kaggle.com → Create → New Notebook
   - Click "Settings" (⚙️) on the right sidebar
   - Set Accelerator to "GPU P100"
   - Enable Internet
   - Save settings

2. Add the ImageNet dataset:
   - Click "Add data" on the right sidebar
   - Search for "ImageNet-1K"
   - Add the dataset to your notebook

## Step 2: Install Dependencies

Copy and paste these commands into the first cell:
```python
# Clone repository
!git clone https://github.com/your-repo/resnet50-training.git
%cd resnet50-training

# Install requirements
!pip install -r requirements.txt

# Verify GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
```

## Step 3: Configure Training

Create a new cell and add:
```python
from config import Config

# Initialize configuration
config = Config()

# Update data path for Kaggle
config.data_dir = '/kaggle/input/imagenet-1k'
config.save_dir = '/kaggle/working/checkpoints'

# Adjust for Kaggle environment
config.batch_size = 128  # Will be auto-adjusted based on GPU memory
config.num_workers = 2   # Kaggle typically allows 2-4 workers
config.use_amp = True    # Enable mixed precision
config.epochs = 100      # Adjust as needed

# Print configuration
config.print_config()
```

## Step 4: Initialize Training

Create a new cell and add:
```python
from trainer import train_imagenet
import sys
import logging

# Setup logging to display in notebook
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Initialize training with error handling
try:
    # Create trainer instance
    trainer = ImageNetTrainer(config)
    
    # Setup training (dataset, model, etc.)
    trainer.setup()
    
    # Start training
    trainer.train()
    
except Exception as e:
    print(f"Error during training: {str(e)}")
    raise
```

## Step 5: Monitor Training Progress

Create a new cell for monitoring (optional):
```python
# Monitor GPU usage
!nvidia-smi

# Display current GPU memory usage
import torch
allocated = torch.cuda.memory_allocated(0) / 1e9
reserved = torch.cuda.memory_reserved(0) / 1e9
print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
```

## Step 6: Save Results

After training completes, create a new cell:
```python
# Zip checkpoints
!zip -r /kaggle/working/checkpoints.zip /kaggle/working/checkpoints

# Zip logs
!zip -r /kaggle/working/logs.zip /kaggle/working/logs

print("Training artifacts saved in '/kaggle/working'")
```

## Common Issues and Solutions

### 1. Out of Memory Errors
If you encounter OOM errors, adjust the configuration:
```python
# Reduce batch size
config.batch_size = 64  # or smaller

# Enable gradient checkpointing
config.use_gradient_checkpointing = True
```

### 2. Slow Training
Monitor and optimize:
```python
# Check data loading
config.num_workers = 1  # Try reducing workers

# Verify GPU utilization
!nvidia-smi -l 1
```

### 3. Dataset Issues
Verify dataset structure:
```python
import os
train_path = os.path.join(config.data_dir, 'train')
val_path = os.path.join(config.data_dir, 'val')

print(f"Training classes: {len(os.listdir(train_path))}")
print(f"Validation classes: {len(os.listdir(val_path))}")
```

## Best Practices for Kaggle

1. **Save Frequently**: Kaggle sessions can timeout
   ```python
   config.save_freq = 1  # Save every epoch
   ```

2. **Monitor Resources**: Keep track of GPU usage
   ```python
   # Add to a separate cell
   !while true; do nvidia-smi; sleep 30; done
   ```

3. **Clean Up**: Remove unnecessary files
   ```python
   !rm -rf /kaggle/working/temp_files
   ```

4. **Backup Important Files**: Save crucial outputs
   ```python
   !cp /kaggle/working/checkpoints/best_model.ckpt /kaggle/working/
   ```

## Complete Example Notebook

Here's a complete notebook structure:

```python
# Cell 1: Setup
!git clone https://github.com/your-repo/resnet50-training.git
%cd resnet50-training
!pip install -r requirements.txt

# Cell 2: Configuration
from config import Config
config = Config()
config.data_dir = '/kaggle/input/imagenet-1k'
config.save_dir = '/kaggle/working/checkpoints'
config.print_config()

# Cell 3: Training
from trainer import train_imagenet
try:
    trainer = ImageNetTrainer(config)
    trainer.setup()
    trainer.train()
except Exception as e:
    print(f"Training error: {str(e)}")

# Cell 4: Save Results
!zip -r /kaggle/working/checkpoints.zip /kaggle/working/checkpoints
!zip -r /kaggle/working/logs.zip /kaggle/working/logs
``` 