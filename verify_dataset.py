import os
import sys
from pathlib import Path
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def verify_imagenet(data_dir: str):
    """Verify ImageNet-1K dataset structure and contents."""
    data_dir = Path(data_dir)
    
    # Check directory structure
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    assert train_dir.exists(), f"Training directory not found at {train_dir}"
    assert val_dir.exists(), f"Validation directory not found at {val_dir}"
    
    print("Checking training set...")
    train_dataset = ImageFolder(train_dir)
    print("Checking validation set...")
    val_dataset = ImageFolder(val_dir)
    
    # Verify number of classes
    assert len(train_dataset.classes) == 1000, \
        f"Expected 1000 training classes, found {len(train_dataset.classes)}"
    assert len(val_dataset.classes) == 1000, \
        f"Expected 1000 validation classes, found {len(val_dataset.classes)}"
    
    # Verify number of images
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    
    assert n_train >= 1281167, \
        f"Expected ≥1,281,167 training images, found {n_train}"
    assert n_val == 50000, \
        f"Expected 50,000 validation images, found {n_val}"
    
    # Verify class consistency
    train_classes = set(train_dataset.classes)
    val_classes = set(val_dataset.classes)
    assert train_classes == val_classes, \
        "Training and validation classes don't match"
    
    # Verify image loading
    print("Verifying image loading...")
    for dataset, name in [(train_dataset, "train"), (val_dataset, "val")]:
        print(f"Testing {name} set...")
        for i in tqdm(range(min(1000, len(dataset)))):  # Test first 1000 images
            try:
                img, label = dataset[i]
            except Exception as e:
                print(f"Error loading image {i} from {name} set: {str(e)}")
                sys.exit(1)
    
    print("\nDataset verification successful!")
    print(f"Training images: {n_train:,}")
    print(f"Validation images: {n_val:,}")
    print(f"Number of classes: {len(train_classes):,}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Verify ImageNet-1K dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to ImageNet dataset')
    args = parser.parse_args()
    
    verify_imagenet(args.data_dir) 