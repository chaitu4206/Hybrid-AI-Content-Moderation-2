import os
import numpy as np
import torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)
import sys

sys.path.insert(
    0,
    r"C:\Users\chait\BERT\src"
)
from preprocessing import LABELS


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# MODEL
# =========================================================

model_path = "../models/bert-toxic-model-v3"

tokenizer = BertTokenizer.from_pretrained(
    model_path
)

model = BertForSequenceClassification.from_pretrained(
    model_path
)

model.to(device)

model.eval()


# =========================================================
# LOAD OPTIMIZED THRESHOLDS
# =========================================================

threshold_path = (
    "../models/evaluation-v3/"
    "best_thresholds.npy"
)

if os.path.exists(threshold_path):

    best_thresholds = np.load(
        threshold_path,
        allow_pickle=True
    ).item()

    print(
        "\nUsing validation-optimized thresholds:"
    )

    for label in LABELS:

        print(
            f"{label:<15}: "
            f"{best_thresholds[label]:.2f}"
        )

else:

    print(
        "\nWarning: optimized thresholds not found."
    )

    print(
        "Using default threshold = 0.50"
    )

    best_thresholds = {
        label: 0.50
        for label in LABELS
    }


# =========================================================
# PREDICTION
# =========================================================

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.sigmoid(
            outputs.logits
        )[0]

    results = {}

    for i, label in enumerate(LABELS):

        probability = float(
            probabilities[i]
        )

        threshold = best_thresholds[
            label
        ]

        results[label] = {
            "probability": round(
                probability,
                4
            ),

            "threshold": round(
                threshold,
                2
            ),

            "prediction": int(
                probability >= threshold
            )
        }

    return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print(
        "\nAI-Powered Content Moderation System"
    )

    print(
        "===================================="
    )

    text = input(
        "\nEnter comment: "
    )

    predictions = predict(
        text
    )

    print(
        "\nToxicity Analysis:\n"
    )

    for label, values in predictions.items():

        status = (
            "YES"
            if values["prediction"]
            else "NO"
        )

        print(
            f"{label:<15} | "
            f"Probability: "
            f"{values['probability']:.4f} | "
            f"Threshold: "
            f"{values['threshold']:.2f} | "
            f"{status}"
        )