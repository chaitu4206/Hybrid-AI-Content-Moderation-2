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
    split_dataset
)

from preprocessing import (
    preprocess_dataframe
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models/spam_bert"

RESULTS_DIR = "results/spam"


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading dataset...")

df = load_dataset(
    max_samples=5000
)

_, _, test_df = split_dataset(
    df
)


# ============================================================
# PREPROCESS
# ============================================================

test_df = preprocess_dataframe(
    test_df
)


# ============================================================
# CONVERT TO HUGGING FACE DATASET
# ============================================================

test_dataset = Dataset.from_pandas(
    test_df[
        ["text", "label"]
    ],
    preserve_index=False
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

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


# ============================================================
# TOKENIZE TEST DATA
# ============================================================

print("Tokenizing test data...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.remove_columns(
    ["text"]
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

trainer = Trainer(
    model=model
)

print("Generating predictions...")

predictions = trainer.predict(
    test_dataset
)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=1
)

true_labels = test_df[
    "label"
].values


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=[
        "ham",
        "spam"
    ],
    digits=4,
    zero_division=0
)

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


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
        "Ham",
        "Spam"
    ]
)

display.plot()

plt.title(
    "SMS Spam Classification - Confusion Matrix"
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

results_df[
    "predicted_label"
] = predicted_labels

results_df[
    "true_class"
] = results_df[
    "label"
].map({
    0: "ham",
    1: "spam"
})

results_df[
    "predicted_class"
] = results_df[
    "predicted_label"
].map({
    0: "ham",
    1: "spam"
})


prediction_file = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)

results_df.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

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