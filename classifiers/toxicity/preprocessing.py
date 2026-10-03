import pandas as pd
import re


LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove usernames
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Keep normal punctuation because BERT can use it
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def load_and_clean_data(path):

    df = pd.read_csv(path)

    print("Columns found in dataset:")
    print(df.columns.tolist())

    # Detect text column
    possible_text_columns = [
        "comment_text",
        "text",
        "comment"
    ]

    text_column = None

    for col in possible_text_columns:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        raise ValueError(
            "No valid text column found."
        )

    print(f"Using text column: {text_column}")

    # Remove missing text
    df = df.dropna(
        subset=[text_column]
    )

    # Remove duplicate comments
    df = df.drop_duplicates(
        subset=[text_column]
    )

    # Check labels
    missing_labels = [
        label
        for label in LABELS
        if label not in df.columns
    ]

    if missing_labels:
        raise ValueError(
            f"Missing labels: {missing_labels}"
        )

    # Clean text
    df["clean_text"] = (
        df[text_column]
        .apply(clean_text)
    )

    # Remove empty comments
    df = df[
        df["clean_text"].str.len() > 0
    ]

    df = df.reset_index(drop=True)

    print("\nData cleaned successfully!")
    print("Dataset shape:", df.shape)

    return df