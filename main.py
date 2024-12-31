from matplotlib import pyplot as plt
import torch
from src.data_loader import load_and_preprocess_data, balance_dataset
from src.tokenizer import load_tokenizer, tokenize_data
from src.dataset import SalesDataset
from src.trainer import train_model
import numpy as np
from transformers import AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder  # Ensure this import is correct
import pandas as pd

# Load and preprocess the training dataset
data = load_and_preprocess_data("data/twitter_training.csv")

# Balance the training dataset
balanced_data = balance_dataset(data)
train_texts = np.array(balanced_data["tweet_content"].values.tolist())
train_labels = np.array(balanced_data["sentiment_label"].values)

print(balanced_data["sentiment_label"].value_counts())
print(balanced_data.head())

# Load and preprocess the validation dataset
validation_data = pd.read_csv("data/twitter_validation.csv")
validation_data.columns = ['tweet_id', 'entity', 'sentiment', 'tweet_content']

label_encoder = LabelEncoder()
validation_data['sentiment_label'] = label_encoder.fit_transform(validation_data['sentiment'])
validation_texts = validation_data['tweet_content'].tolist()
validation_labels = validation_data['sentiment_label'].tolist()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenize the training and validation data
tokenizer = load_tokenizer()
train_tokenized = tokenize_data(train_texts.tolist(), tokenizer)
validation_tokenized = tokenize_data(validation_texts, tokenizer)

train_dataset = SalesDataset(train_tokenized, train_labels)
validation_dataset = SalesDataset(validation_tokenized, validation_labels)

# Compute class weights for imbalanced datasets
class_counts = np.bincount(train_labels)  # Number of occurrences of each class
total_samples = sum(class_counts)
class_weights = 1.0 / class_counts  # Inverse frequency for class weights
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"Class Weights: {class_weights}")

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=4
)

# Train model
trainer = train_model(
    train_dataset=train_dataset,
    val_dataset=validation_dataset,
    model=model,
    output_dir="./model_with_validation",
    class_weights=class_weights,
)

# Evaluate predictions and generate confusion matrix
validation_predictions = trainer.predict(validation_dataset)
preds = np.argmax(validation_predictions.predictions, axis=1)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
conf_matrix = confusion_matrix(validation_labels, preds, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["negative", "neutral", "positive", "irrelevant"])
disp.plot(cmap="Blues")
plt.show()


print("Training with validation dataset complete.")
