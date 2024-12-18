from torch.utils.data import Dataset
import torch

class SalesDataset(Dataset):
    def __init__(self, tokenized_inputs, labels):
        self.input_ids = tokenized_inputs["input_ids"]
        self.attention_mask = tokenized_inputs["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
