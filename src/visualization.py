"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_samples(dataset, num_samples=12, title="Dataset Samples"):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset: PyTorch dataset
        num_samples (int): Number of samples to display
        title (str): Plot title
    """
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    fig.suptitle(title)
    
    for ax in axes.ravel():
        idx = np.random.randint(len(dataset))
        x, y = dataset[idx]
        
        # Handle different label formats
        if isinstance(y, torch.Tensor):
            label = y.item()
        elif isinstance(y, (list, np.ndarray)):
            label = int(y[0]) if len(y) > 0 else int(y)
        else:
            label = int(y)
        
        img = x.squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_predictions(dataset, indices, title="Predictions", max_images=8):
    """
    Visualize predictions for specific indices.
    
    Args:
        dataset: PyTorch dataset
        indices: Tensor or list of indices to visualize
        title (str): Plot title
        max_images (int): Maximum number of images to show
    """
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()
    
    num_images = min(len(indices), max_images)
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    
    if num_images == 1:
        axes = [axes]
    
    fig.suptitle(title)
    
    for ax, idx in zip(axes, indices[:max_images]):
        x, y = dataset[int(idx)]
        
        # Handle different label formats
        if isinstance(y, torch.Tensor):
            label = y.item()
        elif isinstance(y, (list, np.ndarray)):
            label = int(y[0]) if len(y) > 0 else int(y)
        else:
            label = int(y)
        
        img = x.squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """
    Plot training history (loss over epochs).
    
    Args:
        history (dict): Dictionary with 'train_loss' and 'val_loss' lists
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig

