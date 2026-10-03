import sys
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_DIR = "models/hate_speech_bert"


LABEL_NAMES = {
    0: "HATE SPEECH",
    1: "OFFENSIVE LANGUAGE",
    2: "NEITHER"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Hate Speech BERT model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[0][predicted_class]
        .item()
    )

    return (
        LABEL_NAMES[predicted_class],
        confidence,
        probabilities[0].tolist()
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("HATE SPEECH CLASSIFIER")
    print("=" * 60)

    print(
        "\nEnter a sentence to classify."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        text = input(
            "\nText: "
        ).strip()

        if text.lower() == "exit":

            break

        if not text:

            print(
                "Please enter some text."
            )

            continue

        label, confidence, probabilities = (
            predict(text)
        )

        print(
            f"\nPrediction: {label}"
        )

        print(
            f"Confidence: {confidence:.4f}"
        )

        print("\nClass probabilities:")

        print(
            f"Hate Speech: "
            f"{probabilities[0]:.4f}"
        )

        print(
            f"Offensive Language: "
            f"{probabilities[1]:.4f}"
        )

        print(
            f"Neither: "
            f"{probabilities[2]:.4f}"
        )