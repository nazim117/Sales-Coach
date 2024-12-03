from textblob import TextBlob
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the text and returns a sentiment score.
    """
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

def extract_key_phrases(text):
    """
    Extracts noun phrases from the text and returns them.
    """
    blob = TextBlob(text)
    noun_phrases = blob.noun_phrases
    return noun_phrases

def evaluate_sales_performance(text):
    """
    Evaluates sales performance based on specific keywords and phrases in the text.
    """
    positive_sales_phrases = ["buy", "purchase", "deal", "next step", "close", "decision", "product fit"]
    objection_phrases = ["price", "budget", "too expensive", "no need", "not interested"]

    positive_score = sum(phrase in text.lower() for phrase in positive_sales_phrases)
    objection_score = sum(phrase in text.lower() for phrase in objection_phrases)

    return positive_score, objection_score

def generate_summary_and_suggestions(text):
    """
    Generates a summary and suggestions for improvement based on sentiment and sales performance.
    """
    sentiment_score = analyze_sentiment(text)
    noun_phrases = extract_key_phrases(text)
    positive_score, objection_score = evaluate_sales_performance(text)

    summary = f"Summary of the Call:\n"
    summary += f"Sentiment: {'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'}\n"
    summary += f"Key Topics Discussed: {', '.join(noun_phrases)}\n"

    suggestions = f"\nSuggestions for Improvement:\n"
    if positive_score == 0:
        suggestions += "1. Pitching: Ensure to emphasize the benefits and value of the product to the customer.\n"
    if objection_score > 0:
        suggestions += "2. Objection Handling: Address the customer’s concerns, especially around pricing or need. Offer alternatives like payment plans or discounts.\n"
    if sentiment_score < 0:
        suggestions += "3. Sentiment: The conversation had a negative tone. Be more empathetic and listen actively to the customer.\n"
    if positive_score > 0:
        suggestions += "4. Closing: Make sure to ask for the sale or propose a follow-up meeting to keep the momentum.\n"
    
    suggestions += "5. Qualification: Ask more qualifying questions to better understand the customer's needs and budget.\n"
    
    return summary + suggestions
