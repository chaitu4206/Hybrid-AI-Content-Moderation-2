import torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)


MODEL_PATH = "models/violence_bert"


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Loading Violence BERT model...")

tokenizer = BertTokenizer.from_pretrained(
    MODEL_PATH
)

model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)

model.eval()


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

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    confidence = float(
        probabilities[predicted_class]
    )

    return (
        predicted_class,
        confidence,
        probabilities
    )


if __name__ == "__main__":

    print("\n")
    print("=" * 55)
    print("VIOLENCE CLASSIFIER")
    print("=" * 55)

    print(
        "\nEnter text to classify."
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        text = input(
            "\nText: "
        )

        if text.lower() == "exit":
            break

        predicted_class, confidence, probabilities = predict(
            text
        )

        if predicted_class == 1:

            prediction = "VIOLENCE"

        else:

            prediction = "NOT VIOLENCE"

        print(
            f"\nPrediction: {prediction}"
        )

        print(
            f"Confidence: {confidence:.4f}"
        )

        print(
            "\nClass probabilities:"
        )

        print(
            f"NOT VIOLENCE: "
            f"{float(probabilities[0]):.4f}"
        )

        print(
            f"VIOLENCE: "
            f"{float(probabilities[1]):.4f}"
        )