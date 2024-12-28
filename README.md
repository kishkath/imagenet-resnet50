# ResNet50 ImageNet-1K Training

This repository contains code for training ResNet50 from scratch on ImageNet-1K (ILSVRC2012), targeting 70% top-1 accuracy. The implementation uses PyTorch Lightning and supports distributed training, mixed precision, and various modern training techniques.

## Features

- Distributed Data Parallel (DDP) training
- Mixed Precision Training (AMP)
- OneCycleLR scheduling
- WandB integration for experiment tracking
- Modular code structure
- Both command-line and Jupyter notebook interfaces

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

This implementation uses the ImageNet-1K (ILSVRC2012) dataset from Hugging Face, which contains:
- 1000 object classes
- ~1.28 million training images
- 50,000 validation images
- Images are organized in folders by class

### Dataset Setup

1. The dataset will be automatically downloaded from Hugging Face on first use:
```bash
# Download and prepare dataset
python download_dataset.py --data-dir /path/to/imagenet
```

2. After the first download, the dataset will be cached by Hugging Face. Subsequent runs will use the cached version, making setup much faster.

3. By default, the cache is stored in `~/.cache/huggingface/`. For EC2 instances, you might want to change the cache location:
```bash
# Set cache location (before running the download script)
export HF_HOME=/data/huggingface_cache
export HF_DATASETS_CACHE=/data/huggingface_cache/datasets
```

### Dataset Structure
After download and preparation, the dataset will be organized as:
```
/path/to/imagenet/
├── train/          # ~1.28M training images
│   ├── n00000000/  # Class directories
│   ├── n00000001/
│   └── ...         # 1000 class folders total
└── val/            # 50K validation images
    ├── n00000000/
    ├── n00000001/
    └── ...         # 1000 class folders total
```

### Dataset Verification
The download script automatically verifies the dataset, but you can run verification separately:
```bash
python verify_dataset.py --data-dir /path/to/imagenet
```

## Training

### Command Line Interface

To train using the command line interface:

```bash
python train.py --data-dir /path/to/imagenet \
                --batch-size 256 \
                --epochs 100 \
                --workers 8 \
                --lr 0.1 \
                --wandb
```

Additional options:
- `--no-distributed`: Disable distributed training
- `--no-amp`: Disable automatic mixed precision
- `--save-dir`: Path to save checkpoints (default: 'checkpoints')
- `--momentum`: SGD momentum (default: 0.9)
- `--weight-decay`: Weight decay (default: 1e-4)

### Jupyter Notebook

Alternatively, you can use the provided Jupyter notebook `train.ipynb`. Open the notebook and update the configuration parameters according to your setup.

## Training Configuration

The training uses the following key configurations:

- **Optimizer**: SGD with momentum
- **Learning Rate Schedule**: OneCycleLR
  - max_lr: 0.1
  - pct_start: 0.3
  - div_factor: 25.0
  - final_div_factor: 1e4
- **Augmentations**:
  - RandomResizedCrop
  - RandomHorizontalFlip
  - ColorJitter
  - RandomGrayscale
  - GaussianBlur
- **Batch Size**: 256 per GPU
- **Mixed Precision**: Enabled by default
- **Gradient Clipping**: 1.0

## AWS EC2 Setup

For training on AWS EC2:

1. Choose an appropriate instance type (recommended: p3.16xlarge or p4d.24xlarge)
2. Use Deep Learning AMI with PyTorch support
3. Configure security groups and networking
4. Mount ImageNet dataset on a fast storage volume
5. Update the data path in the configuration

Example EC2 launch command:
```bash
aws ec2 run-instances \
    --image-id ami-xxxxxxxx \
    --instance-type p3.16xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxx \
    --subnet-id subnet-xxxxxxxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]'
```

## Monitoring

The training progress can be monitored through:
- WandB dashboard (if enabled)
- Training logs
- Saved checkpoints

## Results

Expected results after training:
- Top-1 Accuracy: ~70% (after ~90-100 epochs)
- Training time: ~3-4 days on 8 V100 GPUs

## License

This project is licensed under the MIT License - see the LICENSE file for details. 