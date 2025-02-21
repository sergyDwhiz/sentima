import json
from pathlib import Path

# Load headlines from a file
def load_headlines():
    try:
        with open('data/headlines.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Save annotated headlines to a file
def save_annotations(annotations):
    Path('data').mkdir(exist_ok=True)
    with open('data/annotated_headlines.json', 'w') as f:
        json.dump(annotations, f, indent=2)

def view_annotations():
    """View all saved annotations"""
    try:
        with open('data/annotated_headlines.json', 'r') as f:
            annotations = json.load(f)
            print("\n=== Saved Annotations ===")
            print(f"Total annotations: {len(annotations)}\n")
            for i, item in enumerate(annotations, 1):
                print(f"{i}. {item['headline']}")
                print(f"   Sentiment: {item['sentiment']}\n")
    except FileNotFoundError:
        print("No saved annotations found.")

def annotate_headlines(headlines):
    """Interactive annotation tool"""
    annotations = []
    try:
        # Load existing annotations if any
        with open('data/annotated_headlines.json', 'r') as f:
            annotations = json.load(f)
            print(f"Loaded {len(annotations)} existing annotations.")
    except FileNotFoundError:
        pass

    print("\nFor each headline, enter:")
    print("1: Positive")
    print("2: Neutral")
    print("3: Negative")
    print("E: Exit and save\n")
    print(f"Starting from annotation #{len(annotations) + 1}")

    try:
        for headline in headlines[len(annotations):]:  # Continue from where we left off
            print(f"\nHeadline {len(annotations) + 1}/{len(headlines)}:")
            print(f"'{headline}'")

            while True:
                choice = input("Sentiment (1/2/3/E)? ").strip().lower()
                if choice == 'e':
                    raise KeyboardInterrupt  # Handle exit gracefully
                if choice in ['1', '2', '3']:
                    sentiment = {
                        '1': 'positive',
                        '2': 'neutral',
                        '3': 'negative'
                    }[choice]

                    annotations.append({
                        'headline': headline,
                        'sentiment': sentiment
                    })
                    save_annotations(annotations)  # Save after each annotation
                    break
                print("Invalid choice! Please try again.")
    except KeyboardInterrupt:
        print("\nSaving progress...")
        save_annotations(annotations)
        print(f"Saved {len(annotations)} annotations.")

def main():
    action = input("Choose action:\n1: Start/Continue Annotation\n2: View Saved Annotations\nChoice (1/2)? ").strip()

    if action == '1':
        headlines = load_headlines()
        if not headlines:
            print("Error: No headlines found! Run scrape_news.py first.")
            return
        annotate_headlines(headlines)
    elif action == '2':
        view_annotations()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()

