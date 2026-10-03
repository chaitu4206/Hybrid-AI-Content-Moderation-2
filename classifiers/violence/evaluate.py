import os
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt

from preprocessing import load_csv


MODEL_PATH = "models/violence_bert"

TEST_PATH = "data/violence/test.csv"

RESULTS_DIR = "results/violence"


def main():

    print("Loading test dataset...")

    test_df = load_csv(
        TEST_PATH
    )

    test_dataset = Dataset.from_pandas(
        test_df
    )

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_PATH
    )

    def tokenize_function(examples):

        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=64
        )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True
    )

    print("Loading trained model...")

    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    trainer = Trainer(
        model=model
    )

    print("Generating predictions...")

    predictions = trainer.predict(
        test_dataset
    )

    logits = predictions.predictions

    predicted_labels = np.argmax(
        logits,
        axis=1
    )

    true_labels = np.array(
        test_df["label"]
    )

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "not_violence",
            "violence"
        ],
        digits=4
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    print("\n")
    print("=" * 60)
    print("VIOLENCE CLASSIFICATION REPORT")
    print("=" * 60)

    print(report)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        f"{RESULTS_DIR}/classification_report.txt",
        "w"
    ) as file:

        file.write(
            "VIOLENCE CLASSIFICATION REPORT\n"
        )

        file.write("=" * 60)
        file.write("\n\n")

        file.write(report)

        file.write(
            f"\nAccuracy: {accuracy:.4f}\n"
        )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        "Violence Classification Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        ["Not Violence", "Violence"]
    )

    plt.yticks(
        [0, 1],
        ["Not Violence", "Violence"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        f"{RESULTS_DIR}/confusion_matrix.png"
    )

    plt.close()

    # --------------------------------------------------
    # PREDICTIONS CSV
    # --------------------------------------------------

    output_df = test_df.copy()

    output_df["predicted_label"] = predicted_labels

    output_df["predicted_class"] = [
        "VIOLENCE"
        if x == 1
        else "NOT_VIOLENCE"
        for x in predicted_labels
    ]

    output_df.to_csv(
        f"{RESULTS_DIR}/test_predictions.csv",
        index=False
    )

    print("\nEvaluation completed.")

    print(
        f"Report: {RESULTS_DIR}/classification_report.txt"
    )

    print(
        f"Confusion matrix: {RESULTS_DIR}/confusion_matrix.png"
    )

    print(
        f"Predictions: {RESULTS_DIR}/test_predictions.csv"
    )


if __name__ == "__main__":
    main()