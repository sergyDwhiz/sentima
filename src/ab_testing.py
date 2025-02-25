from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import json
from datetime import datetime
import numpy as np
from scipy import stats

class ModelVariant:
    def __init__(self, name, model_path):
        self.name = name
        model_path = Path(model_path)

        if not model_path.exists():
            raise RuntimeError(
                f"Model not found at {model_path}. "
                "Please run 'python src/model.py' first to train and optimize the model."
            )

        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model.eval()
        self.predictions = []
        self.latencies = []
        self.confidences = []

    def predict(self, text):
        start_time = time.time()

        with torch.no_grad():
            inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(outputs.logits, dim=1)
            confidence = probabilities[0][prediction].item()

        latency = (time.time() - start_time) * 1000  # Convert to ms
        sentiment = {0: 'positive', 1: 'neutral', 2: 'negative'}[prediction.item()]

        self.predictions.append(sentiment)
        self.latencies.append(latency)
        self.confidences.append(confidence)

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "latency_ms": latency
        }

def run_ab_test(test_data, num_iterations=100):
    """Run A/B test comparing baseline and optimized models"""
    try:
        # Initialize models
        print("Loading baseline model...")
        baseline = ModelVariant("baseline", Path("models/sentiment_model"))

        print("Loading optimized model...")
        optimized = ModelVariant("optimized", Path("models/optimized_model"))

        results = []

        # Run predictions
        for text in test_data:
            baseline_result = baseline.predict(text)
            optimized_result = optimized.predict(text)

            results.append({
                "text": text,
                "baseline": baseline_result,
                "optimized": optimized_result
            })

        # Calculate statistics
        stats_summary = calculate_statistics(baseline, optimized)

        # Save results
        save_results(results, stats_summary)

        return results, stats_summary
    except RuntimeError as e:
        print(f"Error: {str(e)}")
        print("\nPlease follow these steps:")
        print("1. Train the model first: python src/model.py")
        print("2. Make sure both models exist in the 'models' directory")
        print("3. Try running the A/B test again")
        return None, None

def calculate_statistics(baseline, optimized):
    """Calculate statistical significance of results"""
    # Latency comparison
    t_stat, p_value = stats.ttest_ind(baseline.latencies, optimized.latencies)

    # Confidence comparison
    conf_t_stat, conf_p_value = stats.ttest_ind(baseline.confidences, optimized.confidences)

    # Agreement rate
    agreement = sum(b == o for b, o in zip(baseline.predictions, optimized.predictions)) / len(baseline.predictions)

    return {
        "baseline_mean_latency": np.mean(baseline.latencies),
        "optimized_mean_latency": np.mean(optimized.latencies),
        "latency_improvement": ((np.mean(baseline.latencies) - np.mean(optimized.latencies)) / np.mean(baseline.latencies)) * 100,
        "latency_p_value": p_value,
        "baseline_mean_confidence": np.mean(baseline.confidences),
        "optimized_mean_confidence": np.mean(optimized.confidences),
        "confidence_p_value": conf_p_value,
        "model_agreement_rate": agreement
    }

def save_results(results, stats):
    """Save A/B test results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("ab_test_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f"ab_test_{timestamp}.json", 'w') as f:
        json.dump({
            "results": results,
            "statistics": stats
        }, f, indent=2)

def main():
    # Test headlines
    test_data = [
        "Scientists make breakthrough in renewable energy",
        "Stock market crashes amid global concerns",
        "New study reveals neutral findings about climate change",
        "Global economy shows strong growth",
        "Natural disaster leaves thousands homeless"
    ]

    print("Running A/B test...")
    results, stats = run_ab_test(test_data)

    if results and stats:
        print("\nA/B Test Results:")
        print(f"Latency Improvement: {stats['latency_improvement']:.1f}%")
        print(f"P-value (latency): {stats['latency_p_value']:.4f}")
        print(f"Model Agreement Rate: {stats['model_agreement_rate']:.2f}")
        print(f"Mean Confidence - Baseline: {stats['baseline_mean_confidence']:.3f}")
        print(f"Mean Confidence - Optimized: {stats['optimized_mean_confidence']:.3f}")

if __name__ == "__main__":
    main()
