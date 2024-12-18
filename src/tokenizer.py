from transformers import BertTokenizer
import torch

def load_tokenizer(model_name="bert-base-uncased"):
    """
    Loads the tokenizer for a pre-trained model.

    Args:
        model_name (str): Name of the pre-trained model (default: bert-base-uncased).

    Returns:
        tokenizer: Loaded tokenizer instance.
    """
    return BertTokenizer.from_pretrained(model_name)

def tokenize_data(texts, tokenizer, max_len=512):
    # Debugging step
    print(f"Original Text Length: {len(texts[0])}")  # Print first text length
    tokenized = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    print(f"Tokenized Input IDs Length: {len(tokenized['input_ids'][0])}")
    return tokenized
