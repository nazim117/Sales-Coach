from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn.functional as F

model_path = "model_fold_1/checkpoint-7"
tokenizer_path = "bert-base-uncased"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

model.eval()

label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze(0)

        if len(probs) > 1 and probs[1] > 0.35:
            predicted_class = 1
        else:
            predicted_class = torch.argmax(probs).item()

    return label_mapping[predicted_class]

test_texts = [
    "I am extremely happy with the product. It’s fantastic!",  # Expected: positive
    "This is the worst experience I’ve ever had.",            # Expected: negative
    "The product seems ok, but I’m going to think about it",          # Expected: neutral
]

for text in test_texts:
    sentiment = predict_sentiment(text, model, tokenizer)
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
