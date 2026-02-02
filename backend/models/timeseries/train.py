"""
Train LSTM Model for Sales Forecasting
With MLflow experiment tracking
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
import os
import sys

# Fix import path - add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

# Now import from the model file directly
from model import SalesLSTM, create_dataloaders

# Check for MPS (Apple Silicon GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("💻 Using CPU")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(config):
    """
    Main training function
    
    Args:
        config: dictionary with hyperparameters
    """
    
    print("="*60)
    print("🔥 TRAINING LSTM SALES FORECASTING MODEL")
    print("="*60)
    
    # Start MLflow run
    mlflow.set_experiment("sales-forecasting")
    
    with mlflow.start_run(run_name=f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log hyperparameters
        mlflow.log_params(config)
        
        # Load data
        print("\n📊 Loading data...")
        df = pd.read_csv('data/raw/sales_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for one product
        product_id = 'PROD_001'
        df_product = df[df['product_id'] == product_id].sort_values('date')
        sales_data = df_product['sales_quantity'].values
        
        print(f"   Product: {product_id}")
        print(f"   Data points: {len(sales_data)}")
        
        # Create dataloaders
        train_loader, val_loader, scaler = create_dataloaders(
            sales_data,
            sequence_length=config['sequence_length'],
            prediction_horizon=config['prediction_horizon'],
            train_ratio=config['train_ratio'],
            batch_size=config['batch_size']
        )
        
        # Create model
        print("\n🧠 Creating model...")
        model = SalesLSTM(
            input_size=1,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['prediction_horizon'],
            dropout=config['dropout']
        ).to(device)
        
        print(model)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Learning rate scheduler (reduces LR when loss plateaus)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print(f"\n🏋️ Training for {config['num_epochs']} epochs...")
        print("="*60)
        
        # Training loop
        for epoch in range(config['num_epochs']):
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss = validate_epoch(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}] | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'scaler': scaler,
                    'config': config
                }, 'backend/models/saved_models/lstm_best.pt')
        
        print("="*60)
        print(f"✅ Training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        
        # Plot training curves
        print("\n📈 Creating training visualization...")
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Make predictions on validation set
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(y_batch.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Plot predictions vs actuals (first prediction of each sequence)
        plt.subplot(1, 2, 2)
        plt.scatter(actuals[:, 0], predictions[:, 0], alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], 
                [actuals.min(), actuals.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title('Predictions vs Actuals (Day 1)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/processed/training_results.png', dpi=150, bbox_inches='tight')
        print("✅ Saved: data/processed/training_results.png")
        
        # Log artifacts to MLflow
        mlflow.log_artifact('data/processed/training_results.png')
        mlflow.pytorch.log_model(model, "model")
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel saved to: backend/models/saved_models/lstm_best.pt")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment name: sales-forecasting")
        
        return model, scaler, train_losses, val_losses


if __name__ == "__main__":
    # Hyperparameters
    config = {
        'sequence_length': 30,      # Use 30 days to predict
        'prediction_horizon': 7,    # Predict next 7 days
        'hidden_size': 64,          # LSTM hidden units
        'num_layers': 2,            # Number of LSTM layers
        'dropout': 0.2,             # Dropout rate
        'batch_size': 32,           # Batch size
        'num_epochs': 50,           # Training epochs
        'learning_rate': 0.001,     # Learning rate
        'train_ratio': 0.8          # Train/val split
    }
    
    # Train the model
    model, scaler, train_losses, val_losses = train_model(config)
    
    print("\n🎉 Ready to make predictions!")
    print("Next: Test the model on new data")
