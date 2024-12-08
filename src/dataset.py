from torch.utils.data import Dataset

class SalesDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            key: val[idx] for key, val in self.inputs.items()
        }
