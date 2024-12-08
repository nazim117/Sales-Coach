import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the dataset for cross-validation.
    
    Args:
        file_path (str): Path to the dataset CSV file.
    
    Returns:
        tuple: texts, labels, label_classes
    """
    data = pd.read_csv(file_path)
    
    # Clean text
    def clean_text(text):
        text = text.lower().strip()
        text = text.replace("\n", " ")
        return text

    data["transcript"] = data["transcript"].apply(clean_text)

    # Encode sentiment labels
    label_encoder = LabelEncoder()
    data["sentiment_label"] = label_encoder.fit_transform(data["sentiment"])

    # Return processed data
    return data["transcript"], data["sentiment_label"], label_encoder.classes_
