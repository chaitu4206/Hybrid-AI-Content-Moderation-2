import re
import html

from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove Twitter usernames
    text = re.sub(r"@\w+", "@USER", text)

    # Normalize URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "HTTPURL",
        text
    )

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_function(texts):

    cleaned_texts = [
        clean_text(text)
        for text in texts
    ]

    return tokenizer(
        cleaned_texts,
        padding="max_length",
        truncation=True,
        max_length=128
    )


def preprocess_dataframe(df):

    df = df.copy()

    df["text"] = df["text"].apply(clean_text)

    return df