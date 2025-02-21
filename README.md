# Sentiment Analysis Pipeline (Sentima)

## Project Overview
This project implements an end-to-end sentiment analysis pipeline, from data collection to production deployment with A/B testing capabilities.

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

1. Make sure you have Python 3.8+ installed
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Tasks Breakdown

### Task 1: Data Collection ✓
- Implemented news scraper targeting CNN headlines
- Features:
  - Multiple page scraping
  - Smart filtering of unwanted content
  - Headline extraction from various HTML tags
- Status: Completed, collecting 200+ headlines

### Task 2: Sentiment Annotation ✓
- Implemented interactive annotation tool
- Features:
  - Manual sentiment labeling (positive/neutral/negative)
  - Progress saving after each annotation
  - Session resume capability
  - Annotation viewer
- Status: Annotated!

### Task 3: REST API Development ✓
- Implementing sentiment analysis model
- Features:
  - Using DistilBERT for efficient inference
  - Training on our annotated headlines
  - FastAPI deployment planned
- Current Status: Setting up model training

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the news scraper:
   ```bash
   python src/scrape_news.py
   ```
3. Start annotating headlines:
   ```bash
   python src/annotation.py
   ```
   - Choose 1 to start/continue annotation
   - Choose 2 to view saved annotations
   - Use 1/2/3 to mark sentiments
   - 'E' to save and exit

## Usage

Run the model training:
```bash
python src/model.py
```

## Project Structure
- `src/scrape_news.py`: News scraping implementation
- `src/annotation.py`: Sentiment annotation tool
- `src/train_model.py`: Sentiment classification model
- `data/headlines.json`: Scraped headlines
- `data/annotated_headlines.json`: Annotated dataset

## Next Steps
1. Complete model training on annotated data
2. Implement FastAPI endpoints
3. Add input validation and error handling
4. Document API usage

## API Usage

The sentiment analysis API is accessible via HTTP requests on port 3000.

### Endpoints

1. **Health Check**
```bash
GET http://localhost:3000/health

# Sample Response
{
  "status": "healthy",
  "model_loaded": true
}
```

2. **Sentiment Prediction**
```bash
POST http://localhost:3000/predict
Content-Type: application/json

{
  "text": "Scientists make breakthrough in renewable energy"
}

# Sample Response
{
  "text": "Scientists make breakthrough in renewable energy",
  "sentiment": "positive",
  "confidence": 0.8765
}
```

### Example Usage

Using Python requests:
```python
import requests

# Predict sentiment
response = requests.post(
    "http://localhost:3000/predict",
    json={"text": "Scientists make breakthrough in renewable energy"}
)
print(response.json())

# Check health
health = requests.get("http://localhost:3000/health")
print(health.json())
```

Using curl:
```bash
# Health check
curl http://localhost:3000/health

# Predict sentiment
curl -X POST http://localhost:3000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Scientists make breakthrough in renewable energy"}'
```

### Running the API

1. Train the model first:
```bash
python src/model.py
```

2. Start the API server:
```bash
python src/api.py
```

3. Test the API Simulataneously:
```bash
python src/test_api.py
```

