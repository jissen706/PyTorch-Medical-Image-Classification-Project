"""
CNN Model Architecture for Pneumonia Classification
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for binary classification.
    
    Architecture:
    - Two convolutional blocks with max pooling
    - Fully connected classifier with dropout
    
    Args:
        input_channels (int): Number of input channels (1 for grayscale)
        num_classes (int): Number of output classes (2 for binary classification)
        dropout_rate (float): Dropout probability for regularization
    """
    
    def __init__(self, input_channels=1, num_classes=2, dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # [B, 16, 28, 28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 16, 14, 14]
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 14, 14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # [B, 32, 7, 7]
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 32*7*7]
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)  # Binary classification: single output
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, 28, 28]
            
        Returns:
            torch.Tensor: Logits of shape [B, 1]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name='SimpleCNN', **kwargs):
    """
    Factory function to get model by name.
    
    Args:
        model_name (str): Name of the model to instantiate
        **kwargs: Additional arguments to pass to model constructor
        
    Returns:
        nn.Module: Model instance
    """
    models = {
        'SimpleCNN': SimpleCNN,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)

