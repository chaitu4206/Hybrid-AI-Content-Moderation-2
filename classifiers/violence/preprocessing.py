import re
import pandas as pd

from transformers import BertTokenizer


MODEL_NAME = "bert-base-uncased"

LABELS = [
    "not_violence",
    "violence"
]

id2label = {
    0: "NOT_VIOLENCE",
    1: "VIOLENCE"
}

label2id = {
    "NOT_VIOLENCE": 0,
    "VIOLENCE": 1
}


tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME
)


def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def preprocess_function(examples):

    texts = [
        clean_text(text)
        for text in examples["text"]
    ]

    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=64
    )


def load_csv(path):

    df = pd.read_csv(path)

    df["text"] = df["text"].apply(
        clean_text
    )

    df["label"] = df["label"].astype(int)

    return df