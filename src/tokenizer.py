from transformers import BertTokenizer

def load_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)

def tokenize_data(texts, tokenizer, max_len=128):
    tokenized = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokenized