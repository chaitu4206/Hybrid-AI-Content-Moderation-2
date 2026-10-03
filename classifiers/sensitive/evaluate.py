import os
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from dataset import load_sensitive_dataset


MODEL_PATH = (
    "../models/sensitive_bert"
)

RESULTS_DIR = (
    "../results/sensitive"
)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")

data = load_sensitive_dataset(
    max_samples=3000
)

test_dataset = Dataset.from_dict(
    data["test"]
)


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)


# =========================================================
# MODEL
# =========================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


# =========================================================
# PREDICTION
# =========================================================

trainer = Trainer(
    model=model
)

print("\nGenerating predictions...")

output = trainer.predict(
    test_dataset
)

predictions = np.argmax(
    output.predictions,
    axis=-1
)

labels = np.array(
    data["test"]["label"]
)


# =========================================================
# REPORT
# =========================================================

report = classification_report(
    labels,
    predictions,
    target_names=[
        "not_sensitive",
        "sensitive"
    ],
    digits=4
)

print(
    "\n"
    + "=" * 60
)

print(
    "SENSITIVE CONTENT CLASSIFICATION REPORT"
)

print(
    "=" * 60
)

print(report)


# =========================================================
# SAVE RESULTS
# =========================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


cm = confusion_matrix(
    labels,
    predictions
)

np.savetxt(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.txt"
    ),
    cm,
    fmt="%d"
)

print(
    "\nEvaluation completed."
)


