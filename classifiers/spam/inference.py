import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models/spam_bert"


LABEL_NAMES = {
    0: "HAM",
    1: "SPAM"
}


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print(
    "Loading SMS Spam BERT model..."
)

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
        probabilities[0][
            predicted_class
        ].item()
    )

    return (
        LABEL_NAMES[
            predicted_class
        ],
        confidence,
        probabilities[0].tolist()
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("SMS SPAM CLASSIFIER")
    print("=" * 60)

    print(
        "\nEnter an SMS message to classify."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        text = input(
            "\nSMS: "
        ).strip()

        if text.lower() == "exit":

            break

        if not text:

            print(
                "Please enter a message."
            )

            continue

        label, confidence, probabilities = predict(
            text
        )

        print(
            f"\nPrediction: {label}"
        )

        print(
            f"Confidence: {confidence:.4f}"
        )

        print("\nClass probabilities:")

        print(
            f"Ham: {probabilities[0]:.4f}"
        )

        print(
            f"Spam: {probabilities[1]:.4f}"
        )