# AWS EC2 Training Guide for ImageNet ResNet50

This guide provides detailed steps for setting up and running the ResNet50 training on ImageNet-1K (ILSVRC2012) with multi-GPU support.

## 1. EC2 Instance Selection

### Recommended Instance Types:
- **p3.16xlarge**: 8 V100 GPUs, 64 vCPUs, 488 GB RAM
- **p4d.24xlarge**: 8 A100 GPUs, 96 vCPUs, 1152 GB RAM

### AMI Selection:
Use the latest Deep Learning AMI with PyTorch support:
```bash
# Example AMI ID for Deep Learning AMI GPU PyTorch 2.0.1 (Ubuntu 20.04)
ami-0c7217cdde317cfec  # us-east-1 region
```

## 2. Instance Launch

### Using AWS CLI:
```bash
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type p3.16xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxx \
    --subnet-id subnet-xxxxxxxx \
    --block-device-mappings '[
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": 100,
                "VolumeType": "gp3",
                "Iops": 16000,
                "Throughput": 1000
            }
        },
        {
            "DeviceName": "/dev/sdf",
            "Ebs": {
                "VolumeSize": 1000,
                "VolumeType": "gp3",
                "Iops": 16000,
                "Throughput": 1000
            }
        }
    ]'
```

### Security Group Settings:
- Inbound: Allow SSH (port 22)
- Outbound: Allow all traffic

## 3. Instance Setup

### 3.1. Connect to Instance
```bash
ssh -i /path/to/your-key-pair.pem ubuntu@your-instance-ip
```

### 3.2. Mount Additional Volume
```bash
# List available volumes
lsblk

# Create filesystem
sudo mkfs -t ext4 /dev/nvme1n1

# Create mount point
sudo mkdir /data

# Mount volume
sudo mount /dev/nvme1n1 /data

# Add to fstab for persistent mounting
echo '/dev/nvme1n1 /data ext4 defaults 0 0' | sudo tee -a /etc/fstab

# Set permissions
sudo chown -R ubuntu:ubuntu /data
```

### 3.3. Setup Environment
```bash
# Create conda environment
conda create -n imagenet python=3.8
conda activate imagenet

# Clone repository
git clone https://your-repo-url.git
cd your-repo

# Install requirements
pip install -r requirements.txt
```

### 3.4. ImageNet-1K Dataset Setup

The dataset will be downloaded from Hugging Face and cached for future use. This means subsequent runs will use the cached version instead of downloading again.

```bash
# Create dataset directory
mkdir -p /data/imagenet

# Download and prepare dataset
python download_dataset.py --data-dir /data/imagenet

# The script will:
# 1. Download ImageNet-1K from Hugging Face (cached after first download)
# 2. Organize it in the correct directory structure
# 3. Verify the dataset integrity
```

#### Dataset Location and Caching
Hugging Face datasets are cached by default in:
```bash
# Default cache location
~/.cache/huggingface/datasets/

# To change cache location (recommended for EC2):
export HF_HOME=/data/huggingface_cache
export HF_DATASETS_CACHE=/data/huggingface_cache/datasets
```

#### Cache Management
```bash
# View cache size
du -sh ~/.cache/huggingface/datasets/

# Clear cache if needed (not recommended unless space is critical)
rm -rf ~/.cache/huggingface/datasets/*
```

### 3.5. Dataset Verification
The download script automatically verifies the dataset, but you can run verification separately:
```bash
python verify_dataset.py --data-dir /data/imagenet
```

## 4. Training Launch

### 4.1. Single Node Multi-GPU Training
```bash
# Basic training command
python train.py \
    --data-dir /data/imagenet \
    --batch-size 256 \
    --epochs 100 \
    --workers 8 \
    --lr 0.1 \
    --wandb

# For background running with logging
nohup python train.py \
    --data-dir /data/imagenet \
    --batch-size 256 \
    --epochs 100 \
    --workers 8 \
    --lr 0.1 \
    --wandb \
    > training.log 2>&1 &
```

### 4.2. Monitor Training
```bash
# View logs in real-time
tail -f training.log

# Monitor GPU usage
watch nvidia-smi

# Monitor system resources
htop
```

## 5. Cost Management

### 5.1. Instance Pricing (On-Demand, us-east-1)
- p3.16xlarge: ~$24.48/hour
- p4d.24xlarge: ~$32.77/hour

### 5.2. Cost Optimization
- Use Spot Instances for ~70% discount
- Use automatic shutdown script:
```bash
#!/bin/bash
# shutdown_check.py
import os
import time

while True:
    try:
        with open('training.log', 'r') as f:
            last_lines = f.readlines()[-100:]
            if any('val_acc: 0.70' in line for line in last_lines):
                os.system('sudo shutdown -h now')
    except:
        pass
    time.sleep(300)
```

### 5.3. Data Transfer
- Use AWS S3 for dataset storage
- Consider using AWS DataSync for large transfers

## 6. Troubleshooting

### 6.1. Common Issues
1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   ```bash
   python train.py --batch-size 128 --gradient-checkpointing
   ```

2. **Network Issues**
   - Increase number of workers gradually
   - Use local SSD for dataset

3. **Training Crashes**
   - Check CUDA version compatibility
   - Monitor system memory usage

### 6.2. Performance Optimization
1. **Data Loading**
   ```bash
   # Set environment variables for optimal performance
   export CUDA_LAUNCH_BLOCKING=1
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=0
   ```

2. **Memory Usage**
   ```bash
   # Monitor memory usage
   nvidia-smi dmon -s pucvmet -i 0
   ```

## 7. Best Practices

1. **Data Management**
   - Keep dataset on fast SSD storage
   - Use appropriate number of workers (usually 4 per GPU)

2. **Checkpointing**
   - Save checkpoints every epoch
   - Use automatic model upload to S3

3. **Monitoring**
   - Use WandB for experiment tracking
   - Set up CloudWatch alarms for instance metrics

4. **Shutdown Script**
```bash
#!/bin/bash
# save_and_shutdown.sh
aws s3 cp /path/to/checkpoints s3://your-bucket/checkpoints --recursive
sudo shutdown -h now
``` 