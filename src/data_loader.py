import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Clean text
    def clean_text(text):
        text = text.lower().strip()
        text = text.replace("\n", " ")
        return text
    data["transcript"] = data["transcript"].apply(clean_text)

    # Encode sentiment labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    data["sentiment_label"] = label_encoder.fit_transform(data["sentiment"])

    # Add sentiment polarity
    from textblob import TextBlob
    data["sentiment_polarity"] = data["transcript"].apply(lambda x: TextBlob(x).sentiment.polarity)

    return data

def balance_dataset(data):
    """
    Balances the dataset by oversampling underrepresented classes.
    """
    class_counts = data["sentiment_label"].value_counts()
    max_count = class_counts.max()

    # Oversample underrepresented classes
    balanced_data = data.copy()
    for label, count in class_counts.items():
        if count < max_count:
            samples_to_add = data[data["sentiment_label"] == label].sample(max_count - count, replace=True, random_state=42)
            balanced_data = pd.concat([balanced_data, samples_to_add], ignore_index=True)

    return balanced_data
