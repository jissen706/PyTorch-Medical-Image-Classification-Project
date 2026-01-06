"""
Main training script for Pneumonia Classification
"""
import argparse
import yaml
import logging
import torch
import torch.nn as nn
from pathlib import Path

from src.models import get_model
from src.data_loader import load_datasets, create_dataloaders, calculate_class_weights
from src.trainer import train_one_epoch, evaluate, get_predictions
from src.metrics import compute_metrics, find_best_threshold
from src.utils import setup_device, create_directories, save_model
from src.visualization import visualize_samples, plot_training_history


def setup_logging(log_file=None, level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
            logging.StreamHandler()
        ]
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Pneumonia Classification Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_file = config.get('logging', {}).get('log_file', None)
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    setup_logging(log_file, log_level)
    
    # Create necessary directories
    paths = config.get('paths', {})
    create_directories(paths)
    
    # Setup device
    device = setup_device(config['training'].get('device', 'auto'))
    
    # Load datasets
    logging.info("Loading datasets...")
    data_config = config['data']
    train_ds, val_ds, test_ds, info = load_datasets(
        dataset_name=config['data']['dataset_name'],
        data_dir=data_config.get('data_dir', './data'),
        download=True
    )
    
    logging.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    logging.info(f"Labels: {info['label']}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=data_config['batch_size'],
        num_workers=data_config.get('num_workers', 2),
        pin_memory=data_config.get('pin_memory', True),
        device=str(device)
    )
    
    # Calculate class weights
    class_weights = None
    if config['training'].get('use_class_weights', True):
        w0, w1 = calculate_class_weights(train_ds)
        class_weights = (w0, w1)
        logging.info(f"Class weights - Normal: {w0:.4f}, Pneumonia: {w1:.4f}")
    
    # Initialize model
    logging.info("Initializing model...")
    model_config = config['model']
    model = get_model(
        model_name=model_config['name'],
        input_channels=model_config['input_channels'],
        num_classes=model_config['num_classes'],
        dropout_rate=model_config.get('dropout_rate', 0.2)
    ).to(device)
    
    # Setup loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resumed from epoch {start_epoch}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    # Training loop
    num_epochs = config['training']['epochs']
    logging.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, class_weights
        )
        history['train_loss'].append(train_loss)
        
        # Validate
        val_results = evaluate(model, val_loader, criterion, device, class_weights)
        val_loss = val_results['loss']
        history['val_loss'].append(val_loss)
        
        # Compute metrics
        val_metrics = compute_metrics(
            val_results['logits'],
            val_results['labels'],
            threshold=config['evaluation'].get('threshold', 0.5)
        )
        history['val_metrics'].append(val_metrics)
        
        # Log metrics
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}")
        logging.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        logging.info(f"Val Precision: {val_metrics['precision']:.4f}")
        logging.info(f"Val Recall: {val_metrics['recall']:.4f}")
        logging.info(f"Val F1: {val_metrics['f1']:.4f}")
        logging.info(f"TP: {val_metrics['tp']}, FP: {val_metrics['fp']}, "
                    f"FN: {val_metrics['fn']}, TN: {val_metrics['tn']}")
        
        # Save checkpoint
        checkpoint_path = Path(config['paths']['model_save_dir']) / f'checkpoint_epoch_{epoch + 1}.pth'
        save_model(model, str(checkpoint_path), epoch=epoch, metrics=val_metrics)
    
    # Find best threshold on validation set
    logging.info("\nFinding best threshold on validation set...")
    val_probs, val_labels = get_predictions(model, val_loader, device)
    best_threshold, best_f1, best_metrics, threshold_results = find_best_threshold(
        val_probs, val_labels
    )
    logging.info(f"Best threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
    
    # Evaluate on test set
    logging.info("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, criterion, device, class_weights)
    test_metrics = compute_metrics(
        test_results['logits'],
        test_results['labels'],
        threshold=best_threshold
    )
    
    logging.info("Test Set Results:")
    logging.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logging.info(f"Precision: {test_metrics['precision']:.4f}")
    logging.info(f"Recall: {test_metrics['recall']:.4f}")
    logging.info(f"F1: {test_metrics['f1']:.4f}")
    logging.info(f"TP: {test_metrics['tp']}, FP: {test_metrics['fp']}, "
                f"FN: {test_metrics['fn']}, TN: {test_metrics['tn']}")
    
    # Save final model
    final_model_path = Path(config['paths']['model_save_dir']) / 'final_model.pth'
    save_model(model, str(final_model_path), epoch=num_epochs, metrics=test_metrics)
    
    # Save training history
    import json
    history_path = Path(config['paths']['results_dir']) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logging.info("Training completed!")


if __name__ == '__main__':
    main()

