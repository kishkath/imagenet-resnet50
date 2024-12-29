# ResNet50 ImageNet Training

A robust and efficient implementation for training ResNet50 on ImageNet, targeting 70% top-1 accuracy in around 100 epochs.

## Features

- **Efficient Training**
  - Mixed precision training support
  - Gradient checkpointing for memory efficiency
  - Optimized data loading with Albumentations
  - OneCycleLR scheduling for faster convergence

- **Advanced Augmentations**
  - RandAugment for automated augmentation
  - MixUp augmentation support
  - CutMix augmentation support
  - Comprehensive validation transforms

- **Robust Error Handling**
  - Comprehensive error handling across all modules
  - Custom exceptions for training and dataset errors
  - Detailed error messages and logging
  - System resource verification

- **Real-time Monitoring**
  - Console-based progress tracking
  - GPU memory monitoring
  - Training metrics visualization
  - Resource usage tracking

- **Flexible Configuration**
  - Easy-to-modify configuration system
  - Support for distributed training
  - Automatic resource-based adjustments
  - Configuration validation

## Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
albumentations>=1.3.0
pytorch-lightning>=2.0.0
tqdm>=4.65.0
pillow>=9.0.0
pandas>=1.5.0
pyyaml>=6.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download ImageNet-1K dataset
2. Organize the dataset in the following structure:
```
data/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## Training

### Command Line Training

```bash
python train.py
```

### Configuration Options

Key configurations in `config.py`:
- `batch_size`: Training batch size (default: 128)
- `learning_rate`: Initial learning rate (default: 0.1)
- `epochs`: Number of training epochs (default: 100)
- `use_amp`: Enable mixed precision training (default: True)
- `use_autoaugment`: Enable RandAugment (default: True)
- `use_cutmix`: Enable CutMix augmentation (default: True)
- `use_mixup`: Enable MixUp augmentation (default: True)

### Jupyter Notebook

See `train.ipynb` for interactive training and experimentation.

## Monitoring Training

- Real-time console output shows:
  - Training/validation loss and accuracy
  - Learning rate
  - GPU memory usage
  - Estimated time remaining
  - Batch progress

## Error Handling

The implementation includes comprehensive error handling:
- Dataset structure verification
- System resource checks
- GPU memory monitoring
- Automatic backup creation
- Checkpoint management

## Performance Tips

1. Adjust batch size based on available GPU memory
2. Enable mixed precision training for faster execution
3. Use gradient checkpointing for large models
4. Adjust number of workers based on CPU cores
5. Enable autoaugment for better generalization

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 