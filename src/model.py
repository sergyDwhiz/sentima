# Sentiment analysis model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, DistilBertConfig
from sklearn.model_selection import train_test_split
import torch
import json
import numpy as np
from pathlib import Path
from torch.quantization import quantize_dynamic
import time

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
    Optimize the model for faster inference using quantization.
    Quantization reduces model size and improves inference speed by converting
    floating point weights to 8-bit integers.
    """
    print("\nOptimizing model for inference...")

    # Quantize the model to 8-bit integers for faster inference
    print("Quantizing model...")
    start_time = time.time()
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Only quantize linear layers
        dtype=torch.qint8   # Use 8-bit integers
    )
    print(f"Quantization completed in {time.time() - start_time:.2f} seconds")

    # Save the optimized model for production use
    model_path = Path('models/optimized_model')
    model_path.mkdir(exist_ok=True)
    quantized_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return quantized_model

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

# Train sentiment classifier using transformers
def train_model():
    """Train and optimize the sentiment classifier"""
    # Configure a lighter DistilBERT architecture
    config = DistilBertConfig(
        vocab_size=30522,          # Standard vocabulary size
        max_position_embeddings=512,# Maximum sequence length
        num_attention_heads=8,      # Reduced from 12 for efficiency
        num_hidden_layers=4,        # Reduced from 6 for faster inference
        hidden_size=384,           # Reduced from 768 for smaller model
        num_labels=3              # Three sentiment classes
    )

    model = AutoModelForSequenceClassification.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Load and prepare dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # Add batch processing
    batch_size = 16
    train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(X_test, truncation=True, padding=True, return_tensors='pt')

    # Convert to tensors
    train_dataset = torch.utils.data.TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.tensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training logic
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(3): # Train for 3 epochs
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Average loss {total_loss / len(train_loader):.4f}")

    # Evaluate model
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, tokenizer, X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

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

    # Save the model and tokenizer
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    model.save_pretrained(model_path / 'sentiment_model')
    tokenizer.save_pretrained(model_path / 'sentiment_model')

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