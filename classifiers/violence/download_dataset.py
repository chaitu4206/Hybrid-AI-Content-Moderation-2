from datasets import load_dataset
import pandas as pd
import os

print("Downloading violence dataset...")

dataset = load_dataset("kcrl/Violence")

print(dataset)

frames = []

for split in dataset:
    df = dataset[split].to_pandas()

    print(f"{split}: {len(df)} rows")

    frames.append(df[["text", "label"]])

df = pd.concat(frames, ignore_index=True)

# Convert 3-class labels to binary
# 0 = Non-Violence
# 1 = Passive Violence
# 2 = Direct Violence
#
# Final:
# 0 = Not Violent
# 1 = Violent

df["label"] = df["label"].apply(
    lambda x: 0 if int(x) == 0 else 1
)

# Remove empty rows
df = df.dropna(subset=["text", "label"])

# Remove duplicate texts
df = df.drop_duplicates(subset=["text"])

output_path = "data/violence/violence_data.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nDataset saved to:")
print(output_path)

print("\nDataset size:")
print(len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())