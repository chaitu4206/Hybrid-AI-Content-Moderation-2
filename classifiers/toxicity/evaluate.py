import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score
from preprocessing import load_and_clean_data, LABELS
from dataset import ToxicDataset
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = "../models/bert-toxic-model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.to(device)
model.eval()

# Load data
df = load_and_clean_data("../data/train.csv")

# 🔥 Use small subset for fast evaluation
df = df.sample(2000, random_state=42)

encodings = tokenizer(
    df["clean_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=48,
    return_tensors="pt"
)

dataset = ToxicDataset(encodings, df[LABELS])
dataloader = DataLoader(dataset, batch_size=32)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        probs = torch.sigmoid(outputs.logits)
        preds = (probs > 0.5).int()

        all_preds.append(preds.cpu())
        all_labels.append(batch["labels"].cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print("F1 Micro:", f1_score(all_labels, all_preds, average="micro"))
print("F1 Macro:", f1_score(all_labels, all_preds, average="macro"))