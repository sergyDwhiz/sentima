# Get relevant libraries
import re
import requests
from bs4 import BeautifulSoup

# Define fetch function
def fetch_page(url):
    respond = requests.get(url) # Get the page
    respond.raise_for_status() # Check for errors
    return respond.text # Return the text

def is_unwanted(text):
    # Filter headlines with unwanted patterns
    if '/' in text:
        return True
    if text.strip().upper() == "NEWS":
        return True
    if re.search(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d{1,2}, \d{4}\b', text):
        return True
    words = text.split()
    if len(words) == 2 and all(word.istitle() for word in words):
        return True
    return False

# Define extract function
def extract_sentences(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Get headlines in the <headline> tags and <span> tags with the desired attributes
    headlines = soup.find_all('headline')
    spans = soup.find_all('span', class_="container__headline-text", attrs={"data-editable": "headline"})
    if headlines or spans:
        texts = [h.get_text(separator=' ').strip() for h in headlines] + \
                [s.get_text(separator=' ').strip() for s in spans]
    else:
        # Fallback to common heading tags
        texts = [h.get_text(separator=' ').strip() for h in soup.find_all(['h1', 'h2'])]
    headlines_filtered = [t for t in texts if len(t.split()) >= 5 and not is_unwanted(t)]
    return headlines_filtered

# Define main function
def main():
    # Define multiple pages to scrape (example URLs)
    urls = [
        'https://edition.cnn.com',
        'https://edition.cnn.com/world/americas',
        'https://edition.cnn.com/world/africa',
        'https://edition.cnn.com/world/asia',
        'https://edition.cnn.com/politics',
        'https://edition.cnn.com/business',
        'https://edition.cnn.com/health',
        'https://edition.cnn.com/entertainment',
        'https://edition.cnn.com/style',
        'https://edition.cnn.com/travel'
        # Add additional news page URLs as needed
    ]
    all_headlines = []
    for url in urls:
        html = fetch_page(url)
        headlines = extract_sentences(html)
        all_headlines.extend(headlines)

    if len(all_headlines) < 200:
        print(f"Found only {len(all_headlines)} headlines. Try adding more source URLs.")
    else:
        for headline in all_headlines[:200]:
            print(headline)

# Run main function
if __name__ == '__main__':
    main()
