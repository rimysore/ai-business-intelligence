"""
Train BERT Sentiment Analysis Model
PyTorch implementation with experiment tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

from model import SentimentBERT, create_dataloaders

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels


def train_model(config):
    """
    Main training function
    
    Args:
        config: dictionary with hyperparameters
    """
    
    print("="*60)
    print("🔥 TRAINING BERT SENTIMENT ANALYSIS MODEL")
    print("="*60)
    
    # Start MLflow run
    mlflow.set_experiment("sentiment-analysis")
    
    with mlflow.start_run(run_name=f"bert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log hyperparameters
        mlflow.log_params(config)
        
        # Load data
        print("\n📊 Loading reviews data...")
        df = pd.read_csv('data/raw/reviews_data.csv')
        
        print(f"   Total reviews: {len(df)}")
        print(f"   Positive: {(df['sentiment']==1).sum()} ({(df['sentiment']==1).mean()*100:.1f}%)")
        print(f"   Negative: {(df['sentiment']==0).sum()} ({(df['sentiment']==0).mean()*100:.1f}%)")
        
        # Create model
        print("\n🧠 Creating BERT model...")
        sentiment_model = SentimentBERT(
            model_name=config['model_name'],
            num_labels=2,
            max_length=config['max_length']
        )
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            df,
            sentiment_model.tokenizer,
            max_length=config['max_length'],
            test_size=config['test_size'],
            batch_size=config['batch_size']
        )
        
        # Optimizer
        optimizer = optim.AdamW(
            sentiment_model.model.parameters(),
            lr=config['learning_rate']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Training history
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        best_val_acc = 0
        
        print(f"\n🏋️ Training for {config['num_epochs']} epochs...")
        print("="*60)
        
        # Training loop
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
            
            # Train
            train_loss, train_acc = train_epoch(
                sentiment_model.model, train_loader, optimizer, device
            )
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = validate_epoch(
                sentiment_model.model, val_loader, device
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Print progress
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': sentiment_model.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': config
                }, 'backend/models/saved_models/bert_best.pt')
                print(f"  ✅ New best model saved! (Val Acc: {val_acc:.4f})")
        
        print("\n" + "="*60)
        print(f"✅ Training complete!")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
        
        # Final evaluation
        print("\n📊 Final Evaluation on Validation Set:")
        print("\nClassification Report:")
        print(classification_report(
            val_labels, val_preds, 
            target_names=['Negative', 'Positive'],
            digits=4
        ))
        
        # Visualizations
        print("\n📈 Creating visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Loss curves
        axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        axes[0, 1].plot(train_accs, label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(val_accs, label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training & Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_ylabel('True Label')
        axes[1, 0].set_xlabel('Predicted Label')
        
        # Plot 4: Accuracy comparison
        metrics = ['Train Acc', 'Val Acc']
        values = [train_accs[-1], val_accs[-1]]
        colors = ['#4CAF50', '#2196F3']
        axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Final Accuracy Comparison')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(values):
            axes[1, 1].text(i, v + 0.02, f'{v:.4f}', 
                          ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('data/processed/sentiment_training_results.png', 
                   dpi=150, bbox_inches='tight')
        print("✅ Saved: data/processed/sentiment_training_results.png")
        
        # Log artifacts to MLflow
        mlflow.log_artifact('data/processed/sentiment_training_results.png')
        mlflow.pytorch.log_model(sentiment_model.model, "model")
        
        print("\n" + "="*60)
        print("✅ BERT MODEL TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel saved to: backend/models/saved_models/bert_best.pt")
        print(f"Final Validation Accuracy: {best_val_acc*100:.2f}%")
        
        return sentiment_model, train_losses, val_losses, train_accs, val_accs


if __name__ == "__main__":
    # Hyperparameters
    config = {
        'model_name': 'bert-base-uncased',
        'max_length': 128,          # Max tokens per review
        'batch_size': 16,           # Batch size (16 is good for BERT)
        'num_epochs': 3,            # 3 epochs is usually enough for BERT
        'learning_rate': 2e-5,      # Small LR for fine-tuning
        'test_size': 0.2            # 80/20 train/val split
    }
    
    print("Note: BERT training takes longer than LSTM!")
    print("Expected time: ~5-10 minutes on M3 Mac\n")
    
    # Train the model
    model, train_losses, val_losses, train_accs, val_accs = train_model(config)
    
    print("\n🎉 Ready to analyze sentiment on new reviews!")
