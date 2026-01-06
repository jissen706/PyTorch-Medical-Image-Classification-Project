"""
Utility functions
"""
import torch
import logging
import os
from pathlib import Path


def setup_device(device_config='auto'):
    """
    Setup and return the device for training.
    
    Args:
        device_config (str): Device configuration ('auto', 'cuda', or 'cpu')
        
    Returns:
        torch.device: Device object
    """
    if device_config == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    logging.info(f"Using device: {device}")
    return device


def create_directories(paths):
    """
    Create directories if they don't exist.
    
    Args:
        paths (dict): Dictionary of path names to paths
    """
    for name, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Created/verified directory: {path}")


def save_model(model, path, epoch=None, metrics=None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        path (str): Path to save model
        epoch (int): Current epoch
        metrics (dict): Metrics to save with model
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    logging.info(f"Model saved to {path}")


def load_model(model, path):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        path (str): Path to load model from
        
    Returns:
        dict: Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {path}")
    return checkpoint

