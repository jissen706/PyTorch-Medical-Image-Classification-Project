"""
Inference script for pneumonia classification
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from src.models import get_model
from src.utils import setup_device, load_model


def load_image(image_path, transform=None):
    """
    Load and preprocess an image.
    
    Args:
        image_path (str): Path to image file
        transform: Optional transform to apply
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension


def predict(model, image_tensor, device, threshold=0.5):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        threshold: Classification threshold
        
    Returns:
        dict: Prediction results
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probability = torch.sigmoid(logits).item()
        prediction = 1 if probability >= threshold else 0
    
    return {
        'prediction': prediction,
        'probability': probability,
        'class': 'pneumonia' if prediction == 1 else 'normal'
    }


def main():
    parser = argparse.ArgumentParser(description='Pneumonia Classification Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to chest X-ray image')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = setup_device(config['training'].get('device', 'auto'))
    
    # Load model
    model_config = config['model']
    model = get_model(
        model_name=model_config['name'],
        input_channels=model_config['input_channels'],
        num_classes=model_config['num_classes'],
        dropout_rate=model_config.get('dropout_rate', 0.2)
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {args.model}")
    
    # Load and preprocess image
    image_tensor = load_image(args.image)
    
    # Make prediction
    result = predict(model, image_tensor, device, threshold=args.threshold)
    
    # Print results
    print("\n" + "="*50)
    print("Prediction Results:")
    print("="*50)
    print(f"Class: {result['class'].upper()}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Threshold: {args.threshold}")
    print("="*50 + "\n")
    
    return result


if __name__ == '__main__':
    main()

