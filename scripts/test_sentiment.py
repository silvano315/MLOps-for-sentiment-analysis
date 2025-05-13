import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.prediction import predict_sentiment


def main():
    """CLI utility to test sentiment analysis directly."""

    parser = argparse.ArgumentParser(description="Test sentiment analysis")
    parser.add_argument("text", nargs="*", help="Text to analyze")
    parser.add_argument("-f", "--file", help="File containing text to analyze")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args()

    texts = []
    if args.text:
        texts.extend(args.text)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                file_texts = [line.strip() for line in f if line.strip()]
                texts.extend(file_texts)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            sys.exit(1)

    if not texts:
        print("No texts provided for analysis")
        parser.print_help()
        sys.exit(1)

    try:
        results = predict_sentiment(texts)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for i, result in enumerate(results):
                print(f"\nText {i+1}: {result['text']}")
                print(f"Sentiment: {result['sentiment']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print("Probabilities:")
                for label, prob in result["probabilities"].items():
                    print(f"  {label}: {prob:.2f}")
    except Exception as e:
        print(f"Error analyzing text: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
