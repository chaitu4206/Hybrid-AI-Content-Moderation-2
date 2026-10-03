import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://raw.githubusercontent.com/"
    "t-davidson/hate-speech-and-offensive-language/"
    "master/data/labeled_data.csv"
)

DATA_DIR = "data/hate_speech"
DATA_FILE = os.path.join(DATA_DIR, "labeled_data.csv")


LABEL_NAMES = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}


def download_dataset():
    """Download the Davidson hate speech dataset if not already present."""

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print("Downloading Hate Speech dataset...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_FILE, index=False)
        print("Dataset downloaded successfully.")

    return DATA_FILE


def load_dataset(max_samples=None, random_state=42):
    """
    Load and prepare the dataset.

    max_samples:
        Number of samples to use.
        None = use the complete dataset.
    """

    file_path = download_dataset()

    df = pd.read_csv(file_path)

    # Keep only the columns required for classification
    df = df[["tweet", "class"]]

    # Remove missing values
    df = df.dropna()

    # Rename columns
    df = df.rename(
        columns={
            "tweet": "text",
            "class": "label"
        }
    )

    # Convert labels to integers
    df["label"] = df["label"].astype(int)

    # Optional CPU-friendly sampling
    if max_samples is not None and max_samples < len(df):

        # Stratified sampling keeps class proportions
        df, _ = train_test_split(
            df,
            train_size=max_samples,
            stratify=df["label"],
            random_state=random_state
        )

    df = df.reset_index(drop=True)

    return df


def split_dataset(df, random_state=42):

    # 80% train, 10% validation, 10% test

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def print_dataset_statistics(df, name="Dataset"):

    print(f"\n{name} Statistics")
    print("=" * 50)

    print(f"Total samples: {len(df)}")

    print("\nClass distribution:")

    counts = df["label"].value_counts().sort_index()

    for label, count in counts.items():
        label_name = LABEL_NAMES[int(label)]
        percentage = count / len(df) * 100

        print(
            f"{label} - {label_name}: "
            f"{count} ({percentage:.2f}%)"
        )


if __name__ == "__main__":

    df = load_dataset(max_samples=10000)

    print_dataset_statistics(df)

    train_df, val_df, test_df = split_dataset(df)

    print_dataset_statistics(train_df, "Training")
    print_dataset_statistics(val_df, "Validation")
    print_dataset_statistics(test_df, "Testing")