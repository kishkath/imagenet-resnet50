import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
import torch.distributed as dist
import torch
import logging
from tqdm import tqdm
import sys

# Import local modules
from config import Config
from dataset import ImageNetDataset
from model import ResNet50Module

# Configure logging to display to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Custom callback for detailed logging
class DetailedLoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_batch_size = None
        self.val_batch_size = None
        self.current_epoch = 0
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch = trainer.current_epoch
        logger.info(f"\n{'='*80}\nStarting Epoch {self.current_epoch}\n{'='*80}")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 10 == 0:  # Log every 10 batches
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            logger.info(
                f"Epoch {self.current_epoch} | Batch {batch_idx}/{len(trainer.train_dataloader)} | "
                f"Loss: {loss:.4f} | LR: {trainer.optimizers[0].param_groups[0]['lr']:.6f}"
            )
            
    def on_validation_epoch_start(self, trainer, pl_module):
        logger.info(f"\n{'-'*80}\nStarting Validation\n{'-'*80}")
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 5 == 0:  # Log every 5 validation batches
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            logger.info(f"Validation Batch {batch_idx} | Loss: {loss:.4f}")
            
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        logger.info(
            f"\nEpoch {self.current_epoch} Results:\n"
            f"{'='*40}\n"
            f"Training Loss: {metrics.get('train_loss', 0):.4f}\n"
            f"Training Acc: {metrics.get('train_acc', 0):.4f}\n"
            f"{'='*40}\n"
        )

class ImageNetTrainer:
    def __init__(self, config: Config):
        self.config = config
        print("Loaded ImageNetTrainer...")
        logger.info("Initializing ImageNet Trainer...")
        
    def setup(self):
        logger.info("Setting up training...")
        print("setup training set....")
        # Initialize dataset
        dataset = ImageNetDataset(self.config)
        self.train_loader, self.val_loader = dataset.get_dataloaders(
            distributed=self.config.distributed
        )
        logger.info(f"Dataset loaded - Training samples: {len(self.train_loader.dataset)}, "
                   f"Validation samples: {len(self.val_loader.dataset)}")
        
        # Initialize model
        print('Begin to initialize model')
        self.model = ResNet50Module(config=self.config)
        print("model init done..")
        # Enable gradient checkpointing in the model
        if hasattr(self.model.model, 'set_grad_checkpointing'):
            self.model.model.set_grad_checkpointing(enable=True)
            logger.info("Gradient checkpointing enabled via set_grad_checkpointing")
        else:
            # Alternative way to enable gradient checkpointing for ResNet
            for module in self.model.model.modules():
                if isinstance(module, torch.nn.modules.container.Sequential):
                    module.gradient_checkpointing = True
            logger.info("Gradient checkpointing enabled via module setting")
        
        # Setup logging
        if self.config.use_wandb:
            self.logger = WandbLogger(
                project=self.config.project_name,
                log_model=True,
                group=f"resnet50_{self.config.batch_size}_{self.config.learning_rate}"
            )
            logger.info("WandB logging enabled")
        else:
            self.logger = True
        
        # Setup callbacks
        self.callbacks = [
            DetailedLoggingCallback(),
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
        logger.info("Callbacks configured")
        
    def train(self):
        logger.info("Starting training...")
        
        # Setup strategy for single GPU training on Kaggle
        strategy = "auto"
        
        # Initialize trainer with optimized settings for Kaggle
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='gpu',
            devices=1,  # Single GPU for Kaggle
            strategy=strategy,
            precision=16 if self.config.use_amp else 32,  # Simplified precision setting
            callbacks=self.callbacks,
            logger=self.logger,
            log_every_n_steps=1,  # Log every step for detailed monitoring
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            deterministic=False,  # Faster training
            benchmark=True,  # Optimize CUDA kernels
            enable_progress_bar=True,
            enable_model_summary=True,
            detect_anomaly=False,  # Faster training
            val_check_interval=self.config.val_check_interval
        )
        
        logger.info(
            f"Training Configuration:\n"
            f"Batch Size: {self.config.batch_size}\n"
            f"Learning Rate: {self.config.learning_rate}\n"
            f"Mixed Precision: {self.config.use_amp}\n"
            f"Gradient Clipping: {self.config.gradient_clip_val}\n"
            f"Validation Check Interval: {self.config.val_check_interval}"
        )
        
        # Train
        trainer.fit(
            model=self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader
        )
        
def train_imagenet(config: Config):
    print("In train_imagenet...")
    logger.info("Starting ImageNet training process...")
    
    # Set environment variables for optimal performance
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Single GPU for Kaggle

    print("creating trainer instance...")
    # Create trainer instance
    trainer = ImageNetTrainer(config)

    print("Setting up trainer...")
    # Setup training
    trainer.setup()
    print("training start...")
    # Start training
    trainer.train()
    
    logger.info("Training completed!") 
