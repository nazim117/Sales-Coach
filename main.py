from matplotlib import pyplot as plt
import torch
from src.data_loader import load_and_preprocess_data, balance_dataset
from src.tokenizer import load_tokenizer, tokenize_data
from src.dataset import SalesDataset
from src.trainer import train_model
from sklearn.model_selection import KFold
import numpy as np
from transformers import AutoModelForSequenceClassification

from transformers import TrainingArguments, AutoModelForSequenceClassification

# Load and preprocess the dataset
data = load_and_preprocess_data("data/archive/generated_sales_calls.csv")

# Balance the dataset
balanced_data = balance_dataset(data)
texts = np.array(balanced_data["transcript"].values.tolist())
labels = np.array(balanced_data["sentiment_label"].values)

kf = KFold(n_splits=4, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
    print(f"Processing Fold {fold + 1}")

    train_texts = texts[train_idx]
    val_texts = texts[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    # Tokenize texts
    tokenizer = load_tokenizer()
    train_tokenized = tokenize_data(train_texts.tolist(), tokenizer)
    val_tokenized = tokenize_data(val_texts.tolist(), tokenizer)

    train_dataset = SalesDataset(train_tokenized, train_labels)
    val_dataset = SalesDataset(val_tokenized, val_labels)

    # Compute class weights
    class_counts = np.bincount(train_labels)
    total_samples = sum(class_counts)
    class_weights = torch.tensor([total_samples / count for count in class_counts], dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum()
 
    print(f"Class Weights: {class_weights}")

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3
    )

    # Train model
    trainer = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        output_dir=f"./model_fold_{fold + 1}",
        class_weights=class_weights,
    )

    # Evaluate predictions and generate confusion matrix
    val_predictions = trainer.predict(val_dataset)
    preds = np.argmax(val_predictions.predictions, axis=1)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    conf_matrix = confusion_matrix(val_labels, preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["negative", "neutral", "positive"])
    disp.plot(cmap="Blues")
    plt.show()

    print(f"Fold {fold + 1} complete.")
