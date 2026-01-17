"""
Training loop for CNN + KAN model
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import CNN_KAN, CNN_MLP, count_parameters


class Trainer:
    """Trainer class for CNN + KAN model"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device=config.DEVICE,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        num_epochs=config.NUM_EPOCHS,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        # Tracking
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Create checkpoint directory
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = total_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Full training loop"""
        print(f"\nTraining on {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"★ New best model saved! Val Acc: {val_acc:.2f}%")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }
        
        if is_best:
            path = config.BEST_MODEL_PATH
        else:
            path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        return checkpoint['epoch']


def train_model(use_kan=True):
    """Train the model"""
    # Get dataloaders
    train_loader, val_loader, classes = get_dataloaders()
    
    # Create model
    if use_kan:
        model = CNN_KAN()
    else:
        model = CNN_MLP()
    
    # Print model info
    total, trainable = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader)
    history = trainer.train()
    
    return history, classes


if __name__ == "__main__":
    history, classes = train_model(use_kan=True)
