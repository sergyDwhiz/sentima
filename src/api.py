# Deploy optimized model as a REST API
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import torch
import random
import time

app = Flask(__name__)

# Global variables to hold the loaded models and tokenizer
baseline_model = None
optimized_model = None
tokenizer = None
ab_test_enabled = True
ab_test_split = 0.5  # 50-50 split between models

def load_models():
    """
    Load both baseline and optimized models.
    The optimized model uses 8-bit quantization for faster inference.
    """
    global baseline_model, optimized_model, tokenizer

    baseline_path = Path('models/sentiment_model')
    optimized_path = Path('models/optimized_model')

    if not (baseline_path.exists() and optimized_path.exists()):
        raise RuntimeError("Models not found. Please train models first.")

    tokenizer = AutoTokenizer.from_pretrained(optimized_path)
    baseline_model = AutoModelForSequenceClassification.from_pretrained(baseline_path)
    optimized_model = AutoModelForSequenceClassification.from_pretrained(optimized_path)

    baseline_model.eval()
    optimized_model.eval()

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

    # A/B testing logic
    if ab_test_enabled and random.random() < ab_test_split:
        model = baseline_model
        variant = "baseline"
    else:
        model = optimized_model
        variant = "optimized"

    start_time = time.time()

    # Prediction logic
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(outputs.logits, dim=1)
        confidence = probabilities[0][prediction].item()

    latency = (time.time() - start_time) * 1000  # ms
    sentiment = {0: 'positive', 1: 'neutral', 2: 'negative'}[prediction.item()]

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "latency_ms": latency,
        "model_variant": variant
    })

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": baseline_model is not None and optimized_model is not None and tokenizer is not None
    })

# Load models and start server
if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Models loaded successfully!")
    app.run(host='0.0.0.0', port=3000, debug=True)
