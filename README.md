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

## Getting Started
1. Install dependencies:
   ```bash
   pip install requests beautifulsoup4
   ```
2. Run the news scraper:
   ```bash
   python src/scrape_news.py
   ```

## Project Structure
- `src/scrape_news.py`: News scraping implementation

