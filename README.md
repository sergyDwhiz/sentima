# Sentiment Analysis Pipeline (Sentima)

## Project Overview
This project implements an end-to-end sentiment analysis pipeline, from data collection to production deployment with A/B testing capabilities.

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

### Task 3: REST API Development 🔄
- Implementing sentiment analysis model
- Features:
  - Using DistilBERT for efficient inference
  - Training on our annotated headlines
  - FastAPI deployment planned
- Current Status: Setting up model training

## Getting Started
1. Install dependencies:
   ```bash
   pip install requests beautifulsoup4
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

