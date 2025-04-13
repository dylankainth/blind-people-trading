# sentiment.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')  # Download VADER lexicon if not already present

def analyze_sentiment_vader(text):
    """
    Analyzes the sentiment of text using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).

    VADER is particularly good for social media-like text, but works reasonably well for general text too.
    It provides scores for:
    - negative
    - neutral
    - positive
    - compound (overall sentiment, normalized to range [-1, 1])

    Args:
        text (str): The text to analyze.

    Returns:
        dict: Sentiment scores (positive, negative, neutral, compound).
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores

if __name__ == '__main__':
    test_text_positive = "Company XYZ reports record profits and strong growth!"
    test_text_negative = "Stock price of ABC plummets after earnings miss and increased debt."
    test_text_neutral = "Market analysis shows mixed signals for the technology sector."

    sentiment_positive = analyze_sentiment_vader(test_text_positive)
    sentiment_negative = analyze_sentiment_vader(test_text_negative)
    sentiment_neutral = analyze_sentiment_vader(test_text_neutral)

    print(f"Positive Text Sentiment: {sentiment_positive}")
    print(f"Negative Text Sentiment: {sentiment_negative}")
    print(f"Neutral Text Sentiment: {sentiment_neutral}")