import os
from typing import Tuple, Optional

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config

class ImageNetDataset:
    def __init__(self, config: Config):
        self.config = config
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
    def _get_train_transforms(self):
        return A.Compose([
            A.RandomResizedCrop(
                height=self.config.image_size[0],
                width=self.config.image_size[1],
                scale=self.config.train_aug_scale,
                ratio=self.config.train_aug_ratio
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.Normalize(
                mean=self.config.mean,
                std=self.config.std
            ),
            ToTensorV2(),
        ])
    
    def _get_val_transforms(self):
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(self.config.image_size[0], self.config.image_size[1]),
            A.Normalize(
                mean=self.config.mean,
                std=self.config.std
            ),
            ToTensorV2(),
        ])
    
    def get_datasets(self):
        train_path = os.path.join(self.config.data_dir, self.config.train_dir)
        val_path = os.path.join(self.config.data_dir, self.config.val_dir)
        
        train_dataset = TransformedImageFolder(
            root=train_path,
            transform=self.train_transform
        )
        
        val_dataset = TransformedImageFolder(
            root=val_path,
            transform=self.val_transform
        )
        
        return train_dataset, val_dataset
    
    def get_dataloaders(self, distributed: bool = False):
        train_dataset, val_dataset = self.get_datasets()
        
        train_sampler = None
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=True,
            sampler=train_sampler
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader

class TransformedImageFolder(ImageFolder):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image=np.array(image))['image']
        return image, target 