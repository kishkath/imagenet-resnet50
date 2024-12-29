# Running ResNet50 Training on Kaggle

This guide provides detailed instructions for running the ResNet50 ImageNet training code on Kaggle.

## Setup Instructions

### 1. Create a New Notebook
- Start a new notebook on Kaggle
- Enable GPU acceleration (Settings → Accelerator → GPU)
- Enable internet access if needed

### 2. Environment Setup
```python
# Clone the repository
!git clone <repository-url>
%cd <repository-name>

# Install dependencies
!pip install -r requirements.txt
```

### 3. Dataset Access
- Add the ImageNet-1K dataset to your notebook
- Update the data path in `config.py`:
```python
data_dir = '/kaggle/input/imagenet-1k'
```

## Configuration

### 1. Resource-Aware Settings
The code automatically adjusts settings based on available resources:
- Batch size adjustment based on GPU memory
- Worker count optimization based on CPU cores
- Gradient checkpointing for memory efficiency
- Mixed precision training enabled by default

### 2. Key Parameters
```python
from config import Config

config = Config(
    batch_size=128,  # Adjusted automatically if needed
    epochs=100,
    learning_rate=0.1,
    use_amp=True,    # Mixed precision
    num_workers=2    # Kaggle typically allows 2-4 workers
)
```

## Training Process

### 1. Launch Training
```python
from trainer import train_imagenet

# Start training
train_imagenet(config)
```

### 2. Monitor Progress
The training progress is displayed in real-time:
- Loss and accuracy metrics
- Learning rate changes
- GPU memory usage
- Time estimates
- Batch progress

Example output:
```
Epoch 1 | Batch 50/1000 | Loss: 6.4321 | Acc: 0.2345 | LR: 0.0100 | GPU Mem: 14.2GB | Elapsed: 00:05:23 | Remaining: 01:45:12
```

### 3. Resource Monitoring
The code provides built-in resource monitoring:
- GPU memory tracking
- System memory checks
- Disk space monitoring
- Training speed metrics

## Error Handling

The implementation includes robust error handling:
- Dataset verification
- Resource availability checks
- Automatic error recovery
- Detailed error messages

Example error handling:
```python
try:
    train_imagenet(config)
except Exception as e:
    print(f"Training error: {str(e)}")
```

## Performance Optimization

### 1. Memory Management
- Enable gradient checkpointing:
```python
config.use_gradient_checkpointing = True
```

### 2. Training Speed
- Use mixed precision training (enabled by default)
- Optimize batch size for your GPU
- Adjust number of workers based on CPU availability

### 3. Data Augmentation
Available augmentations:
- RandAugment (automated augmentation)
- MixUp
- CutMix
- Standard transforms (flip, crop, etc.)

## Saving Results

### 1. Checkpoints
- Saved automatically in `/kaggle/working/checkpoints/`
- Best models saved based on validation accuracy
- Automatic cleanup of old checkpoints

### 2. Training Logs
- Real-time console output
- Detailed logging to file
- Error logs and warnings

## Troubleshooting

### Common Issues

1. Out of Memory (OOM)
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. Slow Training
   - Check GPU utilization
   - Optimize number of workers
   - Monitor data loading speed

3. Dataset Issues
   - Verify dataset structure
   - Check data paths
   - Monitor data loading errors

### Getting Help
- Check error messages for detailed information
- Review system resource warnings
- Monitor GPU memory usage
- Check training logs for warnings

## Best Practices

1. Resource Management
   - Monitor GPU memory usage
   - Keep track of disk space
   - Watch system memory usage

2. Training Stability
   - Start with default configurations
   - Enable all error checks
   - Monitor validation metrics

3. Performance
   - Use mixed precision training
   - Enable gradient checkpointing
   - Optimize batch size and workers

4. Data Handling
   - Verify dataset before training
   - Monitor data loading speed
   - Check augmentation effects 