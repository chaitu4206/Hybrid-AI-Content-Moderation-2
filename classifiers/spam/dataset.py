import os
import zipfile
import urllib.request
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/spam"

ZIP_FILE = os.path.join(
    DATA_DIR,
    "smsspamcollection.zip"
)

DATA_FILE = os.path.join(
    DATA_DIR,
    "SMSSpamCollection"
)

# Official UCI dataset download
DATA_URL = (
    "https://archive.ics.uci.edu/static/"
    "public/228/sms+spam+collection.zip"
)


LABEL_NAMES = {
    0: "ham",
    1: "spam"
}


# ============================================================
# DOWNLOAD DATASET
# ============================================================

def download_dataset():

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    if not os.path.exists(DATA_FILE):

        print(
            "Downloading SMS Spam Collection..."
        )

        urllib.request.urlretrieve(
            DATA_URL,
            ZIP_FILE
        )

        print(
            "Download completed."
        )

        print(
            "Extracting dataset..."
        )

        with zipfile.ZipFile(
            ZIP_FILE,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                DATA_DIR
            )

        print(
            "Dataset extracted successfully."
        )

    else:

        print(
            "Dataset already exists."
        )

    return DATA_FILE


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    max_samples=None,
    random_state=42
):

    file_path = download_dataset()

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=[
            "label",
            "text"
        ],
        encoding="utf-8"
    )

    df = df.dropna()

    # Convert labels
    df["label"] = df[
        "label"
    ].map({
        "ham": 0,
        "spam": 1
    })

    df = df.dropna()

    df["label"] = (
        df["label"]
        .astype(int)
    )

    # CPU-friendly sampling
    if (
        max_samples is not None
        and max_samples < len(df)
    ):

        df, _ = train_test_split(
            df,
            train_size=max_samples,
            stratify=df["label"],
            random_state=random_state
        )

    return df.reset_index(
        drop=True
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_dataset(
    df,
    random_state=42
):

    # 80% training
    # 10% validation
    # 10% testing

    train_df, temp_df = (
        train_test_split(
            df,
            test_size=0.20,
            stratify=df["label"],
            random_state=random_state
        )
    )

    val_df, test_df = (
        train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df["label"],
            random_state=random_state
        )
    )

    return (
        train_df.reset_index(
            drop=True
        ),
        val_df.reset_index(
            drop=True
        ),
        test_df.reset_index(
            drop=True
        )
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

def print_dataset_statistics(
    df,
    name="Dataset"
):

    print(
        "\n" + name + " Statistics"
    )

    print(
        "=" * 50
    )

    print(
        f"Total samples: {len(df)}"
    )

    print(
        "\nClass distribution:"
    )

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():

        label_name = LABEL_NAMES[
            int(label)
        ]

        percentage = (
            count /
            len(df) *
            100
        )

        print(
            f"{label} - "
            f"{label_name}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# TEST DATASET LOADING
# ============================================================

if __name__ == "__main__":

    df = load_dataset(
        max_samples=5000
    )

    print_dataset_statistics(
        df,
        "Complete Dataset"
    )

    train_df, val_df, test_df = (
        split_dataset(df)
    )

    print_dataset_statistics(
        train_df,
        "Training Dataset"
    )

    print_dataset_statistics(
        val_df,
        "Validation Dataset"
    )

    print_dataset_statistics(
        test_df,
        "Test Dataset"
    )