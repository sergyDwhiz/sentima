# Deploy model as a REST API

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import torch

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer
    model_path = Path('models/sentiment_model')

    if not model_path.exists():
        raise RuntimeError(
            "Model not found. Please train the model first using: python src/model.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

# Home endpoint
@app.route('/')
def home():
    return jsonify({"message": "Sentima API is running"})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Please provide text in the request body"}), 400

    text = request.json['text']

    # Tokenize and predict
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(outputs.logits, dim=1)
        confidence = probabilities[0][prediction].item()

    sentiment = {0: 'positive', 1: 'neutral', 2: 'negative'}[prediction.item()]

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence
    })

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    })

# Load model and start server
if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Model loaded successfully!")
    app.run(host='0.0.0.0', port=3000, debug=True)
