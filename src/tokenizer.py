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

def tokenize_data(texts, labels, tokenizer, max_len=128):
    """
    Tokenizes the text data for use with a transformer model.

    Args:
        texts (list of str): The input text data.
        labels (list of int): The corresponding labels for the text data.
        tokenizer: Pre-trained tokenizer.
        max_len (int): Maximum length of tokenized sequences.

    Returns:
        dict: Tokenized inputs with attention masks and labels.
    """
    inputs = tokenizer(
        list(texts),
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs["labels"] = torch.tensor(labels)
    return inputs
