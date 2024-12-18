import torch
from src.data_loader import load_and_preprocess_data, balance_dataset
from src.tokenizer import load_tokenizer, tokenize_data
from src.dataset import SalesDataset
from src.trainer import train_model
from sklearn.model_selection import KFold
import numpy as np
from transformers import AutoModelForSequenceClassification

# Load and preprocess the dataset
data = load_and_preprocess_data("data/archive/generated_sales_calls.csv")

# Balance the dataset
balanced_data = balance_dataset(data)
print(balanced_data.columns)
print(balanced_data.head())  # Check the structure of the balanced dataset
print(balanced_data["sentiment_label"].value_counts())  # Confirm class balance

# Prepare data for K-Fold
texts = np.array(balanced_data["transcript"].values.tolist())  # Convert to NumPy array
labels = np.array(balanced_data["sentiment_label"].values)     # Convert to NumPy array

kf = KFold(n_splits=4, shuffle=True, random_state=42)

# Define device once, at the beginning of the loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
    print(f"Processing Fold {fold + 1}")

    # Split data
    train_texts = texts[train_idx]  # NumPy array slicing
    val_texts = texts[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    # Tokenize texts
    tokenizer = load_tokenizer()  # Ensure this loads your tokenizer
    print(f"First 5 train_texts: {train_texts[:5]}")  # Debugging step

    train_tokenized = tokenize_data(train_texts.tolist(), tokenizer)  # Convert back to list for tokenization
    val_tokenized = tokenize_data(val_texts.tolist(), tokenizer)

    # Create datasets
    train_dataset = SalesDataset(train_tokenized, train_labels)
    val_dataset = SalesDataset(val_tokenized, val_labels)

    # Compute class weights
    class_counts = balanced_data["sentiment_label"].value_counts()
    print(balanced_data["sentiment_label"].value_counts())

    total_samples = sum(class_counts)
    class_weights = torch.tensor([total_samples / count for count in class_counts]).to(device)
    class_weights = class_weights / class_weights.sum()  # Normalize
    print(f"Class Weights: {class_weights}")

    print(f"Class Counts: {class_counts}")
    print(f"Class Weights: {class_weights}")

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2", num_labels=3, ignore_mismatched_sizes=True
    )

    # Train model
    trainer = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        output_dir=f"./model_fold_{fold + 1}",
        class_weights=class_weights,
    )

    print(f"Fold {fold + 1} complete.")
