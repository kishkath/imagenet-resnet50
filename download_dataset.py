import os
from datasets import load_dataset
from pathlib import Path
import shutil
from tqdm import tqdm

def download_and_prepare_imagenet(data_dir: str):
    """
    Download ImageNet-1K from Hugging Face and organize it in PyTorch format.
    The dataset will be cached by Hugging Face for future use.
    """
    data_dir = Path(data_dir)
    
    print("Loading ImageNet-1K from Hugging Face...")
    # This will download and cache the dataset
    dataset = load_dataset("imagenet-1k", split=['train', 'validation'])
    
    # Create directory structure
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_split(split_dataset, target_dir, split_name):
        print(f"\nPreparing {split_name} split...")
        
        # Create class directories
        class_ids = set(split_dataset['label'])
        for class_id in tqdm(class_ids, desc="Creating class directories"):
            class_dir = target_dir / f"n{class_id:08d}"
            class_dir.mkdir(exist_ok=True)
        
        # Save images to appropriate directories
        for idx in tqdm(range(len(split_dataset)), desc=f"Processing {split_name} images"):
            sample = split_dataset[idx]
            image = sample['image']
            label = sample['label']
            
            # Save image
            class_dir = target_dir / f"n{label:08d}"
            image_path = class_dir / f"image_{idx:08d}.JPEG"
            image.save(image_path)
    
    # Process train split
    setup_split(dataset[0], train_dir, "training")
    
    # Process validation split
    setup_split(dataset[1], val_dir, "validation")
    
    print("\nDataset preparation completed!")
    print(f"Dataset location: {data_dir}")
    print("\nVerifying dataset structure...")
    os.system(f"python verify_dataset.py --data-dir {data_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download ImageNet-1K from Hugging Face')
    parser.add_argument('--data-dir', type=str, default='/data/imagenet',
                      help='Directory to store the dataset')
    args = parser.parse_args()
    
    download_and_prepare_imagenet(args.data_dir) 