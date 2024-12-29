import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from utils import DatasetError

class ImageNetDataset:
    def __init__(self, config):
        try:
            self.config = config
            print("\nInitializing ImageNet dataset...")
            
            # Setup transforms
            self.train_transform = self._get_train_transform()
            self.val_transform = self._get_val_transform()
            
            print("Transforms initialized")
            
        except Exception as e:
            print(f"Error initializing ImageNet dataset: {str(e)}")
            raise
            
    def _get_train_transform(self) -> A.Compose:
        """Get training data transforms with Albumentations"""
        try:
            transform = [
                A.RandomResizedCrop(
                    height=self.config.image_size,
                    width=self.config.image_size,
                    scale=(0.08, 1.0)
                ),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.2,
                        p=0.8
                    ),
                    A.ToGray(p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                ], p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    border_mode=0,
                    p=0.5
                ),
                A.Normalize(
                    mean=self.config.mean,
                    std=self.config.std
                ),
                ToTensorV2()
            ]
                
            return A.Compose(transform)
            
        except Exception as e:
            print(f"Error creating training transforms: {str(e)}")
            raise
            
    def _get_val_transform(self) -> A.Compose:
        """Get validation data transforms with Albumentations"""
        try:
            return A.Compose([
                A.Resize(
                    height=int(self.config.image_size * 1.14),
                    width=int(self.config.image_size * 1.14)
                ),
                A.CenterCrop(
                    height=self.config.image_size,
                    width=self.config.image_size
                ),
                A.Normalize(
                    mean=self.config.mean,
                    std=self.config.std
                ),
                ToTensorV2()
            ])
        except Exception as e:
            print(f"Error creating validation transforms: {str(e)}")
            raise
            
    def _verify_dataset_structure(self) -> bool:
        """Verify the ImageNet dataset structure"""
        try:
            data_dir = Path(self.config.data_dir)
            
            # Check if directories exist
            if not data_dir.exists():
                raise DatasetError(f"Data directory {data_dir} does not exist")
                
            train_dir = data_dir / 'train'
            val_dir = data_dir / 'val'
            
            if not train_dir.exists():
                raise DatasetError(f"Training directory {train_dir} does not exist")
                
            if not val_dir.exists():
                raise DatasetError(f"Validation directory {val_dir} does not exist")
                
            # Check for class directories
            train_classes = list(train_dir.glob('*'))
            val_classes = list(val_dir.glob('*'))
            
            if len(train_classes) != self.config.num_classes:
                raise DatasetError(
                    f"Expected {self.config.num_classes} training classes, "
                    f"found {len(train_classes)}"
                )
                
            if len(val_classes) != self.config.num_classes:
                raise DatasetError(
                    f"Expected {self.config.num_classes} validation classes, "
                    f"found {len(val_classes)}"
                )
                
            print("Dataset structure verified successfully")
            return True
            
        except Exception as e:
            print(f"Error verifying dataset structure: {str(e)}")
            return False
            
    def get_dataloaders(
        self,
        distributed: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
        """Get training and validation dataloaders"""
        try:
            # Verify dataset structure
            if not self._verify_dataset_structure():
                raise DatasetError("Dataset verification failed")
                
            print("Creating dataloaders...")
            
            # Create datasets
            train_dataset = datasets.ImageFolder(
                os.path.join(self.config.data_dir, 'train'),
                transform=transforms.Lambda(
                    lambda x: torch.from_numpy(
                        self.train_transform(image=np.array(x))['image']
                    ).float()
                )
            )
            
            val_dataset = datasets.ImageFolder(
                os.path.join(self.config.data_dir, 'val'),
                transform=transforms.Lambda(
                    lambda x: torch.from_numpy(
                        self.val_transform(image=np.array(x))['image']
                    ).float()
                )
            )
            
            # Create samplers for distributed training
            train_sampler = None
            val_sampler = None
            
            if distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    num_replicas=self.config.world_size,
                    rank=self.config.rank
                )
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset,
                    num_replicas=self.config.world_size,
                    rank=self.config.rank
                )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=(train_sampler is None),
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                sampler=train_sampler,
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                sampler=val_sampler
            )
            
            print(f"Created dataloaders - Training batches: {len(train_loader)}, "
                  f"Validation batches: {len(val_loader)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"Error creating dataloaders: {str(e)}")
            raise
            
    def mixup_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Mixup data augmentation"""
        try:
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
                
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).cuda()
            
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            
            return mixed_x, y_a, y_b, lam
            
        except Exception as e:
            print(f"Error in mixup: {str(e)}")
            return x, y, y, 1.0
            
    def cutmix_data(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """CutMix data augmentation"""
        try:
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1
                
            batch_size = x.size()[0]
            index = torch.randperm(batch_size).cuda()
            
            # Generate random box
            W = x.size()[2]
            H = x.size()[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)
            
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            # Apply cutmix
            x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            y_a, y_b = y, y[index]
            
            return x, y_a, y_b, lam
            
        except Exception as e:
            print(f"Error in cutmix: {str(e)}")
            return x, y, y, 1.0 