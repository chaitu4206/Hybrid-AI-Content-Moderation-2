import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from dataset import (
    load_dataset,
    split_dataset,
    LABEL_NAMES
)

from preprocessing import preprocess_dataframe


MODEL_DIR = "models/hate_speech_bert"

RESULTS_DIR = "results/hate_speech"


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = load_dataset(
    max_samples=10000
)

_, _, test_df = split_dataset(
    df
)

test_df = preprocess_dataframe(
    test_df
)


# ============================================================
# HUGGINGFACE DATASET
# ============================================================

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.remove_columns(
    ["text"]
)


# ============================================================
# LOAD MODEL
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)


# ============================================================
# TRAINER FOR PREDICTION
# ============================================================

trainer = Trainer(
    model=model
)


# ============================================================
# PREDICTIONS
# ============================================================

print("Generating predictions...")

predictions = trainer.predict(
    test_dataset
)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=1
)

true_labels = test_df["label"].values


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=[
        LABEL_NAMES[0],
        LABEL_NAMES[1],
        LABEL_NAMES[2]
    ],
    digits=4,
    zero_division=0
)

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


# Save report
report_file = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Hate Speech",
        "Offensive",
        "Neither"
    ]
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "Hate Speech Classification - Confusion Matrix"
)

plt.tight_layout()

cm_file = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_file,
    dpi=300
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results_df = test_df.copy()

results_df["predicted_label"] = predicted_labels

results_df["true_class"] = (
    results_df["label"]
    .map(LABEL_NAMES)
)

results_df["predicted_class"] = (
    results_df["predicted_label"]
    .map(LABEL_NAMES)
)

prediction_file = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)

results_df.to_csv(
    prediction_file,
    index=False
)


print("\nEvaluation completed.")

print(
    f"Classification report: {report_file}"
)

print(
    f"Confusion matrix: {cm_file}"
)

print(
    f"Predictions: {prediction_file}"
)