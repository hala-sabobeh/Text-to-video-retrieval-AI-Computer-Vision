import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from utils.data_loader import get_dataloader
from models.text_video_model import TextVideoRetrievalModel
from utils.evaluation import evaluate_retrieval


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, logits_per_video, logits_per_text):
        # Create labels (diagonal is positive)
        labels = torch.arange(logits_per_video.shape[0], device=logits_per_video.device)
        
        # Compute loss in both directions
        loss_v = self.cross_entropy(logits_per_video, labels)
        loss_t = self.cross_entropy(logits_per_text, labels)
        
        return (loss_v + loss_t) / 2


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        frames = batch['frames'].to(device)
        texts = batch['description']
        
        # Forward pass
        video_emb, text_emb = model(frames, texts)
        logits_per_video, logits_per_text = model.compute_similarity(video_emb, text_emb)
        
        # Compute loss
        loss = criterion(logits_per_video, logits_per_text)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def main():
    # Configuration
    config = Config()
    
    print("="*60)
    print("TRAINING TEXT-TO-VIDEO RETRIEVAL MODEL")
    print("="*60)
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Embedding dim: {config.EMBEDDING_DIM}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Device: {config.DEVICE}")
    print("="*60)
    
    # Check batch size
    if config.BATCH_SIZE == 1:
        print("\n⚠️  WARNING: BATCH_SIZE=1 will cause zero loss!")
        print("Please change BATCH_SIZE to 16 in config.py")
        input("Press Enter to continue anyway (not recommended)...")
    
    # Create checkpoint directory
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_loader = get_dataloader(config, mode='train')
    
    # Create model
    print("\nCreating model...")
    model = TextVideoRetrievalModel(config).to(config.DEVICE)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = ContrastiveLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    # Training loop
    print("\nStarting training...")
    print("="*60)
    train_losses = []
    best_recall = 0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE, epoch)
        train_losses.append(avg_loss)
        
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} - Loss: {avg_loss:.4f}")
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            print("Evaluating...")
            metrics, _, _, _ = evaluate_retrieval(model, train_loader, config.DEVICE)
            
            print("Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.2f}")
            
            # Save best model
            if metrics['R@1'] > best_recall:
                best_recall = metrics['R@1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                }, os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
                print(f"✓ Saved best model with R@1: {best_recall:.2f}")
        
        # Save checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))
        
        scheduler.step()
        print("-"*60)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, 'training_loss.png'))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Loss plot saved to {config.CHECKPOINT_DIR}/training_loss.png")
    print(f"Best model saved with R@1: {best_recall:.2f}")
    print("\nNext steps:")
    print("  1. python quick_test_retrieval.py")
    print("  2. python retrieval.py --mode evaluate")
    print("  3. python retrieval.py --mode interactive")
    print("="*60)


if __name__ == '__main__':
    main()