"""
LSTM Model for Sales Forecasting
PyTorch implementation with detailed explanations
"""

import torch
import torch.nn as nn
import numpy as np

class SalesLSTM(nn.Module):
    """
    LSTM Neural Network for Time Series Forecasting
    
    Architecture:
    Input → LSTM Layer 1 → LSTM Layer 2 → Fully Connected → Output
    
    Parameters:
    - input_size: Number of features (1 for univariate - just sales)
    - hidden_size: Number of LSTM hidden units
    - num_layers: Number of stacked LSTM layers
    - output_size: Prediction horizon (how many days to predict)
    - dropout: Regularization to prevent overfitting
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 output_size=7, dropout=0.2):
        super(SalesLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        # batch_first=True means input shape is (batch, sequence, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layer to map LSTM output to predictions
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            predictions: Tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        # out shape: (batch_size, sequence_length, hidden_size)
        # h_n, c_n: final hidden and cell states
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        # out[:, -1, :] shape: (batch_size, hidden_size)
        out = self.dropout(out[:, -1, :])
        
        # Pass through fully connected layer
        # predictions shape: (batch_size, output_size)
        predictions = self.fc(out)
        
        return predictions


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for Time Series
    
    Creates sequences of data for training
    Example: 
        Input: [day1, day2, ..., day30] → Output: [day31, day32, ..., day37]
        We use past 30 days to predict next 7 days
    """
    
    def __init__(self, data, sequence_length=30, prediction_horizon=7):
        """
        Args:
            data: numpy array of sales values
            sequence_length: how many past days to use
            prediction_horizon: how many future days to predict
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def __len__(self):
        """Number of samples we can create"""
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        """
        Get one training sample
        
        Returns:
            X: input sequence (past values)
            y: target sequence (future values to predict)
        """
        # Input: sequence of past values
        X = self.data[idx:idx + self.sequence_length]
        
        # Target: sequence of future values
        y = self.data[idx + self.sequence_length:
                     idx + self.sequence_length + self.prediction_horizon]
        
        # Convert to tensors and add feature dimension
        # Shape: (sequence_length, 1) and (prediction_horizon,)
        X = torch.FloatTensor(X).unsqueeze(-1)
        y = torch.FloatTensor(y)
        
        return X, y


def create_dataloaders(data, sequence_length=30, prediction_horizon=7, 
                       train_ratio=0.8, batch_size=32):
    """
    Create train and validation dataloaders
    
    Args:
        data: numpy array of sales data
        sequence_length: input window size
        prediction_horizon: output window size
        train_ratio: proportion of data for training
        batch_size: batch size for training
    
    Returns:
        train_loader, val_loader, scaler
    """
    from sklearn.preprocessing import MinMaxScaler
    
    # Normalize data to [0, 1] range
    # LSTM works better with normalized data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Split into train and validation
    split_idx = int(len(data_normalized) * train_ratio)
    train_data = data_normalized[:split_idx]
    val_data = data_normalized[split_idx:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, sequence_length, prediction_horizon)
    val_dataset = TimeSeriesDataset(val_data, sequence_length, prediction_horizon)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader, scaler


# Test the model
if __name__ == "__main__":
    print("="*60)
    print("🧠 TESTING LSTM MODEL")
    print("="*60)
    
    # Create model
    model = SalesLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=7,
        dropout=0.2
    )
    
    print(f"\n✅ Model created:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    sequence_length = 30
    test_input = torch.randn(batch_size, sequence_length, 1)
    
    print(f"\n🧪 Testing forward pass:")
    print(f"   Input shape: {test_input.shape}")
    
    output = model(test_input)
    print(f"   Output shape: {output.shape}")
    print(f"✅ Forward pass successful!")
    
    print("\n" + "="*60)
    print("✅ Model is ready for training!")
    print("="*60)
