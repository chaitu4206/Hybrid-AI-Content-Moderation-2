import re
import html


def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Replace URLs with a common token
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "HTTPURL",
        text
    )

    # Replace email addresses
    text = re.sub(
        r"\S+@\S+",
        "EMAIL",
        text
    )

    # Replace phone numbers
    text = re.sub(
        r"\+?\d[\d\s\-]{7,}\d",
        "PHONE",
        text
    )

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def preprocess_dataframe(df):

    df = df.copy()

    df["text"] = (
        df["text"]
        .apply(clean_text)
    )

    return df