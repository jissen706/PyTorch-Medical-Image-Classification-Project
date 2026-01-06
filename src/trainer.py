"""
Training and evaluation utilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device, class_weights=None):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        class_weights (tuple): Class weights (w0, w1) for weighted loss
        
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    # Use weighted loss if class weights provided
    if class_weights is not None:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        w0, w1 = class_weights
        w0_tensor = torch.tensor(w0, device=device)
        w1_tensor = torch.tensor(w1, device=device)
    
    for x, y in tqdm(loader, desc='Training', leave=False):
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)
        
        # Forward pass
        logits = model(x)
        
        # Calculate loss
        if class_weights is not None:
            loss_vec = criterion(logits, y)
            weights = torch.where(y == 1, w1_tensor, w0_tensor)
            loss = (loss_vec * weights).mean()
        else:
            loss = criterion(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        num_samples += x.size(0)
    
    return total_loss / num_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_weights=None):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: PyTorch model
        loader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run on
        class_weights (tuple): Class weights for weighted loss
        
    Returns:
        dict: Dictionary containing loss and all predictions/logits
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    all_logits = []
    all_labels = []
    
    # Use weighted loss if class weights provided
    if class_weights is not None:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        w0, w1 = class_weights
        w0_tensor = torch.tensor(w0, device=device)
        w1_tensor = torch.tensor(w1, device=device)
    
    for x, y in tqdm(loader, desc='Evaluating', leave=False):
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)
        
        # Forward pass
        logits = model(x)
        
        # Calculate loss
        if class_weights is not None:
            loss_vec = criterion(logits, y)
            weights = torch.where(y == 1, w1_tensor, w0_tensor)
            loss = (loss_vec * weights).mean()
        else:
            loss = criterion(logits, y)
        
        total_loss += loss.item() * x.size(0)
        num_samples += x.size(0)
        
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return {
        'loss': total_loss / num_samples,
        'logits': all_logits,
        'labels': all_labels
    }


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Get predictions and probabilities from the model.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        device: Device to run on
        
    Returns:
        tuple: (probabilities, labels)
    """
    model.eval()
    probs_list = []
    labels_list = []
    
    for x, y in loader:
        x = x.to(device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)
        
        logits = model(x)
        probs = torch.sigmoid(logits)
        
        probs_list.append(probs.cpu())
        labels_list.append(y.cpu())
    
    return torch.cat(probs_list, dim=0), torch.cat(labels_list, dim=0)

