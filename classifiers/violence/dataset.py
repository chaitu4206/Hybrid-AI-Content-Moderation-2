import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "data/violence/violence_data.csv"

MAX_SAMPLES = 5000


def load_dataset():

    print("Loading Violence Dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n"
            "Please place violence_data.csv in data/violence/"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nOriginal columns:")
    print(df.columns.tolist())

    # Check required columns
    required_columns = ["text", "label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    # Keep only required columns
    df = df[["text", "label"]].copy()

    # Remove missing values
    df = df.dropna()

    # Convert text to string
    df["text"] = df["text"].astype(str)

    # Convert labels to integer
    df["label"] = df["label"].astype(int)

    # Remove invalid labels
    df = df[df["label"].isin([0, 1])]

    # Shuffle
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Limit dataset for CPU
    if len(df) > MAX_SAMPLES:
        df = df.iloc[:MAX_SAMPLES]

    print("\nComplete Dataset Statistics")
    print("=" * 50)

    print(f"Total samples: {len(df)}")

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    # Train / temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    # Validation / test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    # Save splits
    train_df.to_csv(
        "data/violence/train.csv",
        index=False
    )

    val_df.to_csv(
        "data/violence/validation.csv",
        index=False
    )

    test_df.to_csv(
        "data/violence/test.csv",
        index=False
    )

    print("\nDataset Split")
    print("=" * 50)

    print(f"Training:   {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Testing:    {len(test_df)}")

    return train_df, val_df, test_df


if __name__ == "__main__":

    load_dataset()

    print("\nDataset preparation completed.")