"""
Data loading and preprocessing utilities
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO


def get_transforms():
    """
    Get data transformation pipeline.
    
    Returns:
        transforms.Compose: Transformation pipeline
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
    ])


def load_datasets(dataset_name='pneumoniamnist', data_dir='./data', download=True):
    """
    Load train, validation, and test datasets.
    
    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Directory to save/load data
        download (bool): Whether to download dataset if not present
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, info)
    """
    if dataset_name not in INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    info = INFO[dataset_name]
    transform = get_transforms()
    
    # Get dataset class dynamically
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load datasets
    train_ds = DataClass(split='train', transform=transform, download=download, root=data_dir)
    val_ds = DataClass(split='val', transform=transform, download=download, root=data_dir)
    test_ds = DataClass(split='test', transform=transform, download=download, root=data_dir)
    
    return train_ds, val_ds, test_ds, info


def create_dataloaders(train_ds, val_ds, test_ds, batch_size=128, num_workers=2, 
                      pin_memory=True, device='cuda'):
    """
    Create DataLoader objects for train, validation, and test sets.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory (faster GPU transfer)
        device (str): Device type ('cuda' or 'cpu')
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    pin_memory = pin_memory and (device == 'cuda')
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_label_counts(dataset):
    """
    Count labels in a dataset.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        dict: Dictionary mapping label to count
    """
    labels = []
    for i in range(len(dataset)):
        label = dataset[i][1]
        if isinstance(label, torch.Tensor):
            label = label.item()
        elif isinstance(label, (list, np.ndarray)):
            label = label[0] if len(label) > 0 else label
        labels.append(int(label))
    
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def calculate_class_weights(dataset):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        tuple: (weight_class_0, weight_class_1)
    """
    counts = get_label_counts(dataset)
    c0, c1 = counts.get(0, 1), counts.get(1, 1)
    total = c0 + c1
    
    w0 = total / (2 * c0)
    w1 = total / (2 * c1)
    
    return w0, w1

