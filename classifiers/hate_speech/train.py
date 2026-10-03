import os
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from dataset import (
    load_dataset,
    split_dataset,
    print_dataset_statistics,
    LABEL_NAMES
)

from preprocessing import preprocess_dataframe


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

# CPU-friendly experiment
MAX_SAMPLES = 10000

OUTPUT_DIR = "models/hate_speech_bert"

NUM_LABELS = 3

LABEL2ID = {
    "hate_speech": 0,
    "offensive_language": 1,
    "neither": 2
}

ID2LABEL = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = load_dataset(
    max_samples=MAX_SAMPLES
)

print_dataset_statistics(
    df,
    "Complete Dataset"
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df, val_df, test_df = split_dataset(df)

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


# ============================================================
# PREPROCESSING
# ============================================================

train_df = preprocess_dataframe(train_df)
val_df = preprocess_dataframe(val_df)
test_df = preprocess_dataframe(test_df)


# ============================================================
# CONVERT TO HUGGINGFACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# Remove unnecessary text column
train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_df["label"].values,
    minlength=NUM_LABELS
)

class_weights = (
    len(train_df)
    / (NUM_LABELS * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
)

print("\nClass weights:")

for i, weight in enumerate(class_weights):

    print(
        f"{ID2LABEL[i]}: "
        f"{weight:.4f}"
    )


# ============================================================
# MODEL
# ============================================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)


# ============================================================
# CUSTOM TRAINER WITH CLASS-WEIGHTED LOSS
# ============================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.get("labels")

        outputs = model(
            **inputs
        )

        logits = outputs.get("logits")

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )

        loss = loss_function(
            logits.view(-1, NUM_LABELS),
            labels.view(-1)
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(pred):

    predictions = np.argmax(
        pred.predictions,
        axis=1
    )

    labels = pred.label_ids

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    micro_precision, micro_recall, micro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="micro",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    save_total_limit=1,

    report_to="none",

    use_cpu=True
)


# ============================================================
# TRAINER
# ============================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 60)
print("STARTING HATE SPEECH BERT TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating on test dataset...")

test_results = trainer.evaluate(
    test_dataset
)

print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

for key, value in test_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


print("\nTraining completed successfully.")
print(f"Model saved to: {OUTPUT_DIR}")