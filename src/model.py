# Sentiment analysis model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.model_selection import train_test_split
import torch
import json
import numpy as np
from pathlib import Path

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

# Train sentiment classifier using transformers
def train_model():
    model_name = "distilbert-base-uncased" # Light version of BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )

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

    # Save the model and tokenizer
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    model.save_pretrained(model_path / 'sentiment_model')
    tokenizer.save_pretrained(model_path / 'sentiment_model')

    return model, tokenizer

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