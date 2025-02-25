# Sentiment analysis model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, DistilBertConfig
from sklearn.model_selection import train_test_split
import torch
import json
import numpy as np
from pathlib import Path
from torch.nn import Parameter
import torch.nn.functional as F
import time
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup

# Load annotated headlines
def load_dataset():
    """Load annotated headlines"""
    try:
        with open('data/annotated_headlines.json', 'r') as f:
            data = json.load(f)
            # Access the 'headlines' array in the JSON
            headlines = data.get('headlines', [])
            if not headlines:
                raise ValueError("No headlines found in the JSON file")

            texts = [item['headline'] for item in headlines]
            labels = [{'positive': 0, 'neutral': 1, 'negative': 2}[item['sentiment'].lower()] for item in headlines]
            return train_test_split(texts, labels, test_size=0.2, random_state=42)

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def evaluate_model(model, tokenizer, test_texts, test_labels):
    """Evaluate the model on test data"""
    model.eval()
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')
    test_dataset = torch.utils.data.TensorDataset(
        test_encodings['input_ids'],
        test_encodings['attention_mask'],
        torch.tensor(test_labels)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for a single text"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
        return {0: 'positive', 1: 'neutral', 2: 'negative'}[prediction.item()]

def optimize_model(model, tokenizer):
    """
    Optimize the model for faster inference using weight pruning
    and reduced precision where supported.
    """
    print("\nOptimizing model for inference...")
    start_time = time.time()

    # Prune less important weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                # More aggressive pruning for efficiency
                threshold = param.data.std() * 0.2
                mask = torch.abs(param.data) > threshold
                param.data.mul_(mask.float())

                # Quantize remaining weights to 8-bit precision
                if 'attention' in name or 'intermediate' in name:
                    max_val = torch.max(torch.abs(param.data))
                    param.data = torch.round(param.data / max_val * 127) * max_val / 127

    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")

    # Save the optimized model
    model_path = Path('models/optimized_model')
    model_path.mkdir(exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model

def benchmark_model(model, tokenizer, text, num_iterations=100):
    """
    Measure model inference speed by running multiple predictions
    and calculating average time per prediction.

    Args:
        model: The model to benchmark
        tokenizer: Tokenizer for text preprocessing
        text: Sample text for prediction
        num_iterations: Number of predictions to average over
    """
    total_time = 0

    # Warm up the model to ensure accurate timing
    for _ in range(5):
        _ = predict_sentiment(text, model, tokenizer)

    # Run benchmark iterations
    for _ in range(num_iterations):
        start_time = time.time()
        _ = predict_sentiment(text, model, tokenizer)
        total_time += time.time() - start_time

    # Calculate average time in milliseconds
    avg_time = (total_time / num_iterations) * 1000
    return avg_time

class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def augment_data(texts, labels):
    """Augment training data with simple text transformations"""
    augmented_texts = []
    augmented_labels = []

    for text, label in zip(texts, labels):
        # Original text
        augmented_texts.append(text)
        augmented_labels.append(label)

        # Remove punctuation
        clean_text = ''.join(c for c in text if c.isalnum() or c.isspace())
        augmented_texts.append(clean_text)
        augmented_labels.append(label)

        # Simple word reordering (no NLTK dependency)
        words = text.split()
        if len(words) > 3:
            # Reverse second half of sentence
            mid = len(words) // 2
            shuffled = words[:mid] + words[mid:][::-1]
            augmented_texts.append(' '.join(shuffled))
            augmented_labels.append(label)

            # Add lowercase version
            augmented_texts.append(text.lower())
            augmented_labels.append(label)

    return augmented_texts, augmented_labels

def train_model():
    """Train and optimize the sentiment classifier"""
    # Use proven BERT-base configuration
    config = DistilBertConfig.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3,
        hidden_dropout_prob=0.2,
        attention_dropout=0.2
    )

    print("Loading pretrained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        config=config
    )
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    print("Loading and augmenting dataset...")
    X_train, X_test, y_train, y_test = load_dataset()
    X_train, y_train = augment_data(X_train, y_train)

    print(f"Training data size: {len(X_train)} (after augmentation)")
    print(f"Test data size: {len(X_test)}")

    # Create datasets with progress indicators
    print("Preparing datasets...")
    train_dataset = HeadlineDataset(X_train, y_train, tokenizer)
    test_dataset = HeadlineDataset(X_test, y_test, tokenizer)

    # Training parameters
    batch_size = 16
    epochs = 8
    warmup_steps = 100
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_accuracy = 0
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        accuracy = evaluate_model(model, tokenizer, X_test, y_test)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Average loss: {total_loss / len(train_loader):.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save best model
            model_path = Path('models')
            model_path.mkdir(exist_ok=True)
            model.save_pretrained(model_path / 'sentiment_model')
            tokenizer.save_pretrained(model_path / 'sentiment_model')

    print(f"\nBest validation accuracy: {best_accuracy:.4f}")

    # Benchmark original model performance
    print("\nBenchmarking original model...")
    orig_speed = benchmark_model(
        model,
        tokenizer,
        "Scientists make breakthrough in renewable energy"
    )
    print(f"Original model average inference time: {orig_speed:.2f}ms")

    # Optimize and benchmark the optimized model
    optimized_model = optimize_model(model, tokenizer)
    print("\nBenchmarking optimized model...")
    opt_speed = benchmark_model(
        optimized_model,
        tokenizer,
        "Scientists make breakthrough in renewable energy"
    )
    print(f"Optimized model average inference time: {opt_speed:.2f}ms")
    print(f"Speed improvement: {((orig_speed - opt_speed) / orig_speed) * 100:.1f}%")

    return optimized_model, tokenizer

# Main function
def main():
    print("Training sentiment analysis model...")
    model, tokenizer = train_model()

    # Test prediction
    test_headlines = [
        "Scientists make breakthrough in renewable energy",
        "Stock market crashes amid global concerns",
        "New study reveals neutral findings about climate change"
    ]

    print("\nTesting predictions:")
    for headline in test_headlines:
        sentiment = predict_sentiment(headline, model, tokenizer)
        print(f"Headline: {headline}")
        print(f"Predicted sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()