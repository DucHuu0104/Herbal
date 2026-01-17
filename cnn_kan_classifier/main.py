"""
Main entry point for CNN + KAN Image Classification
"""
import argparse
import random
import numpy as np
import torch

import config
from train import train_model


def set_seed(seed=config.SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="CNN + KAN Image Classification")
    parser.add_argument('--model', type=str, default='kan', choices=['kan', 'mlp'],
                        help='Model type: kan or mlp (default: kan)')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help=f'Number of epochs (default: {config.NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help=f'Learning rate (default: {config.LEARNING_RATE})')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights')
    parser.add_argument('--seed', type=int, default=config.SEED,
                        help=f'Random seed (default: {config.SEED})')
    
    args = parser.parse_args()
    
    # Update config with command line args
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.FREEZE_BACKBONE = args.freeze_backbone
    
    # Set seed
    set_seed(args.seed)
    
    # Print configuration
    print("=" * 60)
    print("CNN + KAN Image Classification")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {'CNN + KAN' if args.model == 'kan' else 'CNN + MLP'}")
    print(f"Dataset: {config.DATA_DIR}")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Backbone frozen: {config.FREEZE_BACKBONE}")
    print("=" * 60)
    
    # Train
    use_kan = args.model == 'kan'
    history, classes = train_model(use_kan=use_kan)
    
    # Save training history
    import json
    history_path = f"{config.CHECKPOINT_DIR}/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
