import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_PATH = (
    "../models/sensitive_bert"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Loading Sensitive Content BERT..."
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)

model.eval()


LABELS = [
    "NOT SENSITIVE",
    "SENSITIVE"
]


def predict(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    prediction = int(
        torch.argmax(probabilities)
    )

    return {
        "label": LABELS[prediction],
        "confidence": float(
            probabilities[prediction]
        ),
        "probabilities": {
            LABELS[i]: float(
                probabilities[i]
            )
            for i in range(2)
        }
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 55
    )

    print(
        "SENSITIVE CONTENT CLASSIFIER"
    )

    print(
        "=" * 55
    )

    print(
        "\nType 'exit' to stop."
    )

    while True:

        text = input(
            "\nText: "
        )

        if text.lower() == "exit":
            break

        result = predict(
            text
        )

        print(
            f"\nPrediction: "
            f"{result['label']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.4f}"
        )

        print(
            "\nClass probabilities:"
        )

        for label, probability in (
            result["probabilities"].items()
        ):

            print(
                f"{label}: "
                f"{probability:.4f}"
            )