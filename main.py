from src.data_loader import load_and_preprocess_data
from src.tokenizer import load_tokenizer, tokenize_data
from src.dataset import SalesDataset
from src.trainer import load_model, train_model
from sklearn.model_selection import KFold
import numpy as np

# Load and preprocess the dataset
texts, labels, label_classes = load_and_preprocess_data("data/archive/generated_sales_calls.csv")

# Ensure texts and labels are NumPy arrays for compatibility with KFold
texts = np.array(texts)
labels = np.array(labels)

# Initialize tokenizer
tokenizer = load_tokenizer()

# Set up K-Fold cross-validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)

# Start cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(texts)):
    print(f"Processing Fold {fold + 1}...")

    # Create train and validation splits using NumPy indexing
    train_texts, val_texts = texts[train_index], texts[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]

    # Tokenize the data
    train_inputs = tokenize_data(train_texts, train_labels, tokenizer)
    val_inputs = tokenize_data(val_texts, val_labels, tokenizer)

    # Create datasets
    train_dataset = SalesDataset(train_inputs)
    val_dataset = SalesDataset(val_inputs)

    # Load the model
    model = load_model(num_labels=len(label_classes))

    # Train the model
    trainer = train_model(train_dataset, val_dataset, model, tokenizer)

    # Save the model for this fold
    model.save_pretrained(f"./sales_coach_model_fold_{fold + 1}")
    tokenizer.save_pretrained(f"./sales_coach_model_fold_{fold + 1}")

    print(f"Fold {fold + 1} complete.")
