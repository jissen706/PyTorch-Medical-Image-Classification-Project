"""
Evaluation metrics and utilities
"""
import torch
import numpy as np


def compute_metrics(logits, labels, threshold=0.5):
    """
    Compute classification metrics from logits.
    
    Args:
        logits (torch.Tensor): Model logits
        labels (torch.Tensor): True labels
        threshold (float): Classification threshold
        
    Returns:
        dict: Dictionary containing metrics
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    labels = labels.long()
    
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def metrics_at_threshold(probs, labels, threshold):
    """
    Compute metrics at a specific threshold.
    
    Args:
        probs (torch.Tensor): Predicted probabilities
        labels (torch.Tensor): True labels
        threshold (float): Classification threshold
        
    Returns:
        tuple: (accuracy, precision, recall, f1, fp, fn)
    """
    preds = (probs >= threshold).long()
    labels = labels.long()
    
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    
    return accuracy, precision, recall, f1, fp, fn


def find_best_threshold(probs, labels, threshold_range=None):
    """
    Find the best threshold based on F1 score.
    
    Args:
        probs (torch.Tensor): Predicted probabilities
        labels (torch.Tensor): True labels
        threshold_range (list): Range of thresholds to test
        
    Returns:
        tuple: (best_threshold, best_f1, metrics_dict)
    """
    if threshold_range is None:
        threshold_range = [i/100 for i in range(5, 61, 5)]
    
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = None
    
    results = {}
    
    for thr in threshold_range:
        acc, prec, rec, f1, fp, fn = metrics_at_threshold(probs, labels, thr)
        results[thr] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'fp': fp,
            'fn': fn
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr
            best_metrics = results[thr]
    
    return best_threshold, best_f1, best_metrics, results

