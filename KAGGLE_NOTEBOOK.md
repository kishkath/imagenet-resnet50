# Running ResNet50 Training in Kaggle Notebook

## Step 1: Setup Kaggle Environment

1. Create a new notebook in Kaggle:
   - Go to https://www.kaggle.com/
   - Click on "Create" → "New Notebook"
   - In notebook settings (right sidebar):
     - Enable GPU (TPU not required)
     - Set accelerator to "GPU P100"
     - Set internet access to "ON"

2. Install required packages:
```python
!pip install pytorch-lightning
!pip install torchmetrics
!pip install seaborn
```

## Step 2: Clone the Repository

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
!cd YOUR_REPO_NAME

# Install requirements
!pip install -r requirements.txt
```

## Step 3: Prepare Dataset

1. For ImageNet dataset:
```python
# Download ImageNet dataset using the provided script
!python download_dataset.py
```

2. For custom dataset:
   - Upload your dataset to Kaggle dataset
   - Mount it in your notebook:
```python
# Example for custom dataset
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('your-dataset-name', path='./data')
!unzip ./data/your-dataset.zip -d ./data
```

## Step 4: Configure Training

Create a configuration file or set parameters directly:

```python
from model import Config, ResNet50Module
import pytorch_lightning as pl

# Create config with three-phase learning rate schedule
config = Config(
    num_classes=1000,  # Adjust based on your dataset
    learning_rate=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    max_lr=0.1,
    epochs=90,
    pct_start=0.3,        # First 30% for warmup phase
    div_factor=25.0,      # Initial lr = max_lr/div_factor
    final_div_factor=1e4, # Final lr = initial_lr/final_div_factor
    three_phase=True,     # Enable three-phase learning
    anneal_strategy='cos' # Cosine annealing
)

# Create model
model = ResNet50Module(config)

# Print learning rate schedule info
print(f"Initial learning rate: {config.max_lr/config.div_factor:.6f}")
print(f"Maximum learning rate: {config.max_lr:.6f}")
print(f"Final learning rate: {config.max_lr/(config.div_factor*config.final_div_factor):.6f}")
print(f"Warmup epochs: {int(config.epochs * config.pct_start)}")
print(f"Annealing epochs: {config.epochs - int(config.epochs * config.pct_start)}")
```

## Step 5: Setup Data and Training

```python
from dataset import ImageNetDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Create dataset
dataset = ImageNetDataset(config)
train_loader, val_loader = dataset.get_dataloaders()

# Setup checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='resnet50-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    monitor='val_loss'
)

# Setup LR monitoring
lr_monitor = LearningRateMonitor(logging_interval='step')

# Create trainer with learning rate monitoring
trainer = Trainer(
    max_epochs=config.epochs,
    accelerator='gpu',
    devices=1,
    precision=16,  # Use mixed precision for faster training
    callbacks=[checkpoint_callback, lr_monitor],
    enable_progress_bar=True
)
```

## Step 6: Start Training

```python
# Start training
trainer.fit(model, train_loader, val_loader)
```

## Step 7: Monitor Training

The training progress will be displayed in:
1. Progress bar showing current epoch and batch
2. Real-time metrics in the notebook output
3. Logs directory containing detailed training logs
4. Plots directory containing visualization charts

To view training plots during or after training:
```python
from visualize import visualize_training

# Get the latest log file
import glob
latest_log = max(glob.glob('logs/training_*.log'), key=os.path.getctime)
visualize_training(latest_log)

# Plot learning rate schedule
import matplotlib.pyplot as plt
import numpy as np

def plot_lr_schedule(config):
    steps = np.linspace(0, config.epochs, 1000)
    total_steps = len(steps)
    warmup_steps = int(total_steps * config.pct_start)
    
    # Calculate learning rates
    lr = []
    for step in range(total_steps):
        if step < warmup_steps:
            # Warmup phase
            lr_step = config.max_lr/config.div_factor + \
                     step * (config.max_lr - config.max_lr/config.div_factor) / warmup_steps
        else:
            # Annealing phase
            pct = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * pct))
            lr_step = config.max_lr * cosine_decay
        lr.append(lr_step)
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, lr)
    plt.title('Three-Phase Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig('plots/lr_schedule.png')
    plt.close()

plot_lr_schedule(config)
```

## Step 8: Save Results

```python
# Save trained model
trainer.save_checkpoint("final_model.ckpt")

# Download results locally
from IPython.display import FileLink
FileLink("final_model.ckpt")
FileLink("plots/training_metrics_latest.png")
FileLink("plots/training_summary_latest.txt")
FileLink("plots/lr_schedule.png")
```

## Common Issues and Solutions

1. **Out of Memory (OOM) Error**:
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training (already enabled)

2. **Slow Training**:
   - Ensure GPU is properly enabled
   - Check GPU utilization: `!nvidia-smi`
   - Optimize number of workers in dataloader

3. **Dataset Issues**:
   - Verify dataset structure
   - Check class labels
   - Ensure proper data normalization

4. **Learning Rate Issues**:
   - Monitor the learning rate curve
   - Adjust pct_start for different warmup duration
   - Tune div_factor and final_div_factor if needed

## Best Practices

1. **Learning Rate Schedule**:
   - Monitor the learning rate progression
   - Ensure warmup phase is sufficient
   - Check if final learning rate is appropriate

2. **Save Checkpoints Regularly**:
   - Model checkpoints are saved automatically
   - Keep best 3 models based on validation loss

3. **Monitor Resources**:
   - Watch GPU memory usage
   - Check disk space regularly
   - Monitor training metrics

4. **Visualize Results**:
   - Training/validation loss curves
   - Accuracy progression
   - Learning rate changes

5. **Save Experiment Results**:
   - Download model checkpoints
   - Save training plots
   - Export metrics to CSV

## Additional Tips

1. **Three-Phase Learning Rate Schedule**:
   - Warmup phase: Gradually increases LR
   - Peak phase: Maintains max LR
   - Annealing phase: Gradually decreases LR

2. **Kaggle Time Limits**:
   - Notebooks run up to 9 hours
   - Save checkpoints frequently
   - Use notebook versioning

3. **Data Management**:
   - Use Kaggle datasets for large files
   - Cache preprocessed data
   - Clean up temporary files

4. **Debugging**:
   - Check logs in `logs` directory
   - Monitor system resources
   - Use smaller subset for testing

5. **Performance Optimization**:
   - Use appropriate batch size
   - Optimize number of workers
   - Enable mixed precision training 