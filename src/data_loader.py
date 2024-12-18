import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import nlpaug.augmenter.word as naw  # Import the augmentation library
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def load_and_preprocess_data(file_path):
    """
    Loads, cleans, augments, and preprocesses the dataset.

    Args:
        file_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Preprocessed dataset with augmented 'transcript' and 'sentiment_label'.
    """
    data = pd.read_csv(file_path)

    # Clean text
    def clean_text(text):
        text = text.lower().strip()
        text = text.replace("\n", " ")
        return text

    data["transcript"] = data["transcript"].apply(clean_text)

    # Apply data augmentation
    print("Performing data augmentation...")
    aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)
    # Apply data augmentation and flatten lists
    print("Performing data augmentation...")
    aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)
    augmented_texts = []

    for text in data["transcript"]:
        augmented_output = aug.augment(text)
        if isinstance(augmented_output, list):
            # Flatten by taking the first augmented output (you can customize this behavior)
            augmented_texts.append(" ".join(augmented_output))
        else:
            augmented_texts.append(augmented_output)
    
    # Duplicate data with augmented texts
    augmented_data = data.copy()
    augmented_data["transcript"] = augmented_texts

    # Combine original and augmented data
    data = pd.concat([data, augmented_data], axis=0).reset_index(drop=True)
    print(f"Dataset size after augmentation: {len(data)}")

    # Encode sentiment labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    data["sentiment_label"] = label_encoder.fit_transform(data["sentiment"])
    label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    print("Label Classes:", label_mapping)
    return data

def balance_dataset(data):
    """
    Balances the dataset using SMOTE for text data, retaining the original text column.
    """
    # Extract features and labels
    X = data["transcript"].values  # Raw text data
    y = data["sentiment_label"]

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
    X_tfidf = vectorizer.fit_transform(X).toarray()

    # Apply SMOTE to the numerical data
    smote = SMOTE(random_state=42, k_neighbors=2)  # Adjust k_neighbors based on your dataset
    X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

    # Get the indices of the resampled data to retrieve corresponding transcripts
    indices = smote.fit_resample(np.arange(len(X)).reshape(-1, 1), y)[0].flatten()

    # Create a new DataFrame with resampled transcripts and labels
    resampled_data = pd.DataFrame(X_resampled, columns=vectorizer.get_feature_names_out())
    resampled_data["transcript"] = data["transcript"].iloc[indices].reset_index(drop=True)
    resampled_data["sentiment_label"] = y_resampled

    return resampled_data

load_and_preprocess_data("data/archive/generated_sales_calls.csv")
