# Test the API endpoints using the requests library.

import requests
import json

def test_api():
    # API endpoint
    BASE_URL = "http://localhost:3000"

    # Test health endpoint
    health_response = requests.get(f"{BASE_URL}/health")
    print("\nHealth Check:")
    print(json.dumps(health_response.json(), indent=2))

    # Test prediction endpoint
    test_headlines = [
        "Scientists make breakthrough in renewable energy",
        "Stock market crashes amid global concerns",
        "New study reveals neutral findings about climate change"
    ]

    print("\nTesting predictions:")
    for headline in test_headlines:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": headline}
        )
        result = response.json()
        print(f"\nHeadline: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    test_api()
