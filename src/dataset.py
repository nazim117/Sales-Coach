from torch.utils.data import Dataset
import torch

class SalesDataset(Dataset):
    def __init__(self, tokenized_inputs, labels):
        """
        Initialize the dataset with tokenized inputs and labels.
        Args:
            tokenized_inputs (dict): A dictionary containing tokenized input IDs and attention masks.
            labels (list or torch.Tensor): Labels corresponding to the inputs.
        """
        self.input_ids = tokenized_inputs["input_ids"]
        self.attention_mask = tokenized_inputs["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are long integers

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Return a single sample from the dataset."""
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }