"""
Dataset loading and preprocessing for herb image classification
"""
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

import config


class HerbDataset(Dataset):
    """Custom dataset for herb image classification"""
    
    def __init__(self, data_dirs, transform=None):
        """
        Args:
            data_dirs: Single directory path or list of directory paths
            transform: Optional transforms
        """
        if isinstance(data_dirs, str):
            self.data_dirs = [data_dirs]
        else:
            self.data_dirs = data_dirs
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and labels from all directories"""
        # Get all class folders from first directory (assume same classes in all dirs)
        first_dir = self.data_dirs[0]
        class_folders = sorted([
            d for d in os.listdir(first_dir) 
            if os.path.isdir(os.path.join(first_dir, d))
        ])
        
        self.classes = class_folders
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_folders)}
        
        # Load samples from ALL directories
        for data_dir in self.data_dirs:
            for class_name in class_folders:
                class_dir = os.path.join(data_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                class_idx = self.class_to_idx[class_name]
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(use_strong_aug=True):
    """Get train and validation transforms
    
    Args:
        use_strong_aug: If True, use stronger augmentation (RandAugment, etc.)
    """
    if use_strong_aug:
        # Strong augmentation with RandAugment
        train_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),  # Resize larger
            transforms.RandomCrop((config.IMAGE_SIZE, config.IMAGE_SIZE)),  # Random crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),  # Stronger rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandAugment(num_ops=2, magnitude=9),  # RandAugment
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),  # Random Erasing
        ])
    else:
        # Basic augmentation
        train_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


def get_dataloaders():
    """Create train and validation dataloaders"""
    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Load full dataset from ALL directories
    full_dataset = HerbDataset(config.DATA_DIRS, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = total_size - train_size
    
    # Split indices
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets with appropriate transforms
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Dataset loaded: {total_size} images, {len(full_dataset.classes)} classes")
    print(f"Train: {train_size} images | Val: {val_size} images")
    print(f"Classes: {full_dataset.classes}")
    
    return train_loader, val_loader, full_dataset.classes


class TransformSubset(Dataset):
    """Subset with custom transform"""
    
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


if __name__ == "__main__":
    # Test dataloader
    train_loader, val_loader, classes = get_dataloaders()
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
