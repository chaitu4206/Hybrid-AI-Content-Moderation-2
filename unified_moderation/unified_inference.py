import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# MODEL PATHS
# ============================================================

MODELS = {

    "toxicity":
        r"C:\Users\chait\BERT\models\bert-toxic-model-v3",

    "sensitive":
        r"C:\Users\chait\models\sensitive_bert",

    "hate_speech":
        r"C:\Users\chait\hatespeech\models\hate_speech_bert",

    "spam":
        r"C:\Users\chait\spam\models\spam_bert",

    "violence":
        r"C:\Users\chait\violence\models\violence_bert"
}


# ============================================================
# LABELS
# ============================================================

LABELS = {

    "sensitive": {
        0: "NOT SENSITIVE",
        1: "SENSITIVE"
    },

    "hate_speech": {
        0: "HATE SPEECH",
        1: "OFFENSIVE LANGUAGE",
        2: "NEITHER"
    },

    "spam": {
        0: "HAM",
        1: "SPAM"
    },

    "violence": {
        0: "NON-VIOLENT",
        1: "VIOLENT"
    }
}


# ============================================================
# LOAD MODELS
# ============================================================

models = {}
tokenizers = {}


def load_model(name, path):

    print(f"\nLoading {name} model...")

    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForSequenceClassification.from_pretrained(
        path
    )

    model.to(DEVICE)
    model.eval()

    tokenizers[name] = tokenizer
    models[name] = model

    print(f"{name} model loaded.")


for name, path in MODELS.items():

    if not os.path.exists(path):

        print(
            f"\nWARNING: Model not found: {path}"
        )

        continue

    load_model(
        name,
        path
    )


# ============================================================
# BINARY / MULTI-CLASS PREDICTION
# ============================================================

def predict_standard(name, text):

    tokenizer = tokenizers[name]
    model = models[name]

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

    predicted_class = torch.argmax(
        probabilities
    ).item()

    confidence = float(
        probabilities[predicted_class]
    )

    return {
        "label":
            LABELS[name][predicted_class],

        "class":
            predicted_class,

        "confidence":
            confidence,

        "probabilities":
            probabilities.cpu().tolist()
    }


# ============================================================
# TOXICITY PREDICTION
# ============================================================

def predict_toxicity(text):

    tokenizer = tokenizers["toxicity"]
    model = models["toxicity"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {
        key: value.to(DEVICE)
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

    # Toxicity model has multiple labels.
    # Read the label names directly from model config.

    for i, label in enumerate(
        model.config.id2label.values()
    ):

        results[label] = float(
            probabilities[i]
        )

    return results


# ============================================================
# COMPLETE MODERATION
# ============================================================

def moderate(text):

    results = {}

    # Toxicity
    if "toxicity" in models:

        results["toxicity"] = predict_toxicity(
            text
        )

    # Sensitive
    if "sensitive" in models:

        results["sensitive"] = predict_standard(
            "sensitive",
            text
        )

    # Hate speech
    if "hate_speech" in models:

        results["hate_speech"] = predict_standard(
            "hate_speech",
            text
        )

    # Spam
    if "spam" in models:

        results["spam"] = predict_standard(
            "spam",
            text
        )

    # Violence
    if "violence" in models:

        results["violence"] = predict_standard(
            "violence",
            text
        )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(text, results):

    print("\n")
    print("=" * 70)
    print("UNIFIED AI CONTENT MODERATION")
    print("=" * 70)

    print("\nInput:")
    print(text)

    print("\n" + "-" * 70)

    # Toxicity
    if "toxicity" in results:

        print("\n[1] TOXICITY")

        for label, probability in (
            results["toxicity"].items()
        ):

            print(
                f"{label:<25}: "
                f"{probability:.4f}"
            )

    # Sensitive
    if "sensitive" in results:

        r = results["sensitive"]

        print("\n[2] SENSITIVE CONTENT")

        print(
            f"Prediction : {r['label']}"
        )

        print(
            f"Confidence : {r['confidence']:.4f}"
        )

    # Hate speech
    if "hate_speech" in results:

        r = results["hate_speech"]

        print("\n[3] HATE SPEECH")

        print(
            f"Prediction : {r['label']}"
        )

        print(
            f"Confidence : {r['confidence']:.4f}"
        )

    # Spam
    if "spam" in results:

        r = results["spam"]

        print("\n[4] SPAM")

        print(
            f"Prediction : {r['label']}"
        )

        print(
            f"Confidence : {r['confidence']:.4f}"
        )

    # Violence
    if "violence" in results:

        r = results["violence"]

        print("\n[5] VIOLENCE")

        print(
            f"Prediction : {r['label']}"
        )

        print(
            f"Confidence : {r['confidence']:.4f}"
        )

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("HYBRID AI CONTENT MODERATION SYSTEM")
    print("=" * 70)

    print("\nAll available BERT classifiers have been loaded.")

    print("\nType 'exit' to stop.")

    while True:

        text = input(
            "\nEnter content: "
        ).strip()

        if text.lower() == "exit":

            print("\nExiting...")
            break

        if not text:

            print("Please enter some text.")
            continue

        results = moderate(
            text
        )

        display_results(
            text,
            results
        )