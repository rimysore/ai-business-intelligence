"""
BERT Sentiment Analysis Model
PyTorch implementation (more stable than TensorFlow version)
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SentimentBERT:
    """
    BERT-based Sentiment Classifier
    
    Uses pre-trained BERT and fine-tunes it for sentiment analysis
    
    Architecture:
    Input Text → BERT Tokenizer → BERT Model → Classification Head → Sentiment
    """
    
    def __init__(self, model_name='bert-base-uncased', num_labels=2, max_length=128):
        """
        Args:
            model_name: Pre-trained BERT model to use
            num_labels: Number of classes (2 for binary sentiment)
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device
        
        # Load tokenizer
        print(f"📦 Loading tokenizer: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Load pre-trained BERT model
        print(f"🧠 Loading BERT model: {model_name}")
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)
        
        print(f"✅ Model loaded on: {device}")
    
    def tokenize_texts(self, texts):
        """
        Convert text to BERT input format
        
        Args:
            texts: List of text strings or single string
        
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # BERT tokenization
        # Converts text to: [CLS] word1 word2 ... [SEP]
        # - [CLS]: Classification token (used for final prediction)
        # - [SEP]: Separator token
        # - Padding: Make all sequences same length
        
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encoded
    
    def predict(self, texts):
        """
        Make predictions on new texts
        
        Args:
            texts: List of review texts or single text
        
        Returns:
            predictions: Array of predicted labels (0 or 1)
            probabilities: Array of confidence scores
        """
        self.model.eval()
        
        # Tokenize
        encoded = self.tokenize_texts(texts)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()


class ReviewDataset(Dataset):
    """
    Custom Dataset for sentiment analysis
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(df, tokenizer, max_length=128, 
                      test_size=0.2, batch_size=16):
    """
    Create train and validation dataloaders from DataFrame
    
    Args:
        df: DataFrame with 'review_text' and 'sentiment' columns
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        test_size: Validation split ratio
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review_text'].tolist(),
        df['sentiment'].tolist(),
        test_size=test_size,
        random_state=42,
        stratify=df['sentiment']  # Keep same class distribution
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training samples: {len(train_texts)}")
    print(f"   Validation samples: {len(val_texts)}")
    
    # Create datasets
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# Test the model
if __name__ == "__main__":
    print("="*60)
    print("🧠 TESTING BERT SENTIMENT MODEL")
    print("="*60)
    
    # Create model
    sentiment_model = SentimentBERT(
        model_name='bert-base-uncased',
        num_labels=2,
        max_length=128
    )
    
    print("\n✅ Model created!")
    total_params = sum(p.numel() for p in sentiment_model.model.parameters())
    trainable_params = sum(p.numel() for p in sentiment_model.model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test tokenization and prediction
    test_texts = [
        "This product is amazing! I love it!",
        "Terrible quality. Very disappointed.",
        "It's okay, nothing special."
    ]
    
    print("\n🧪 Testing predictions (untrained model):")
    predictions, probabilities = sentiment_model.predict(test_texts)
    
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = prob[pred] * 100
        print(f"\nText: {text}")
        print(f"Prediction: {sentiment} ({confidence:.1f}% confidence)")
        print(f"Probabilities: Neg={prob[0]:.3f}, Pos={prob[1]:.3f}")
    
    print("\n" + "="*60)
    print("✅ Model is ready for training!")
    print("="*60)
    print("\nNote: Predictions are random right now (untrained model)")
    print("After training, accuracy will be 90%+!")
