from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the trained model and tokenizer
model_path = "./sales_coach_model_fold_1"  # Path to one of your saved models
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Define the label mapping (adjust if necessary)
label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text, model, tokenizer):
    """
    Predicts the sentiment of a given text using the trained model.

    Args:
        text (str): The input text.
        model: The trained model instance.
        tokenizer: The tokenizer used during training.

    Returns:
        str: The predicted sentiment label.
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Move inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Map numerical prediction to sentiment label
    return label_mapping[predicted_class]

test_texts = [
    "I am extremely happy with the product. It’s fantastic!",  # Expected: positive
    "This is the worst experience I’ve ever had.",            # Expected: negative
    "The product seems okay, but I’m not sure yet.",          # Expected: neutral
]

for text in test_texts:
    sentiment = predict_sentiment(text, model, tokenizer)
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")

