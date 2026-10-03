from datasets import load_dataset
from collections import Counter


MAX_SAMPLES = 3000


def load_sensitive_dataset(max_samples=MAX_SAMPLES):

    print("Loading X-Sensitive dataset...")

    dataset = load_dataset(
        "cardiffnlp/x_sensitive"
    )

    print("\nOriginal dataset:")
    print(dataset)

    processed = {}

    for split in ["train", "validation", "test"]:

        if split not in dataset:
            continue

        data = dataset[split]

        texts = []
        labels = []

        for row in data:

            text = str(row["text"]).strip()

            if not text:
                continue

            # X-Sensitive is multi-label.
            # Any sensitive category = 1.
            row_labels = row["labels"]

            is_sensitive = (
                len(row_labels) > 0
            )

            texts.append(text)
            labels.append(
                1 if is_sensitive else 0
            )

        # CPU-friendly subset
        if len(texts) > max_samples:
            texts = texts[:max_samples]
            labels = labels[:max_samples]

        processed[split] = {
            "text": texts,
            "label": labels
        }

        print(
            f"\n{split.upper()} samples: "
            f"{len(texts)}"
        )

        print(
            "Class distribution:"
        )

        counts = Counter(labels)

        print(
            f"Not Sensitive (0): "
            f"{counts[0]}"
        )

        print(
            f"Sensitive (1): "
            f"{counts[1]}"
        )

    return processed


if __name__ == "__main__":

    data = load_sensitive_dataset()

    print(
        "\nDataset loaded successfully."
    )