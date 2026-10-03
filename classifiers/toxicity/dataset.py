import torch


class ToxicDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings

        if hasattr(labels, "values"):
            self.labels = labels.values
        else:
            self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx]
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.float
        )

        return item

    def __len__(self):
        return len(self.labels)