import os
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from dataset import load_sensitive_dataset


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "bert-base-uncased"

OUTPUT_DIR = (
    "../models/sensitive_bert"
)

MAX_SAMPLES = 3000


# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")

data = load_sensitive_dataset(
    max_samples=MAX_SAMPLES
)


train_dataset = Dataset.from_dict(
    data["train"]
)

validation_dataset = Dataset.from_dict(
    data["validation"]
)

test_dataset = Dataset.from_dict(
    data["test"]
)


# =========================================================
# TOKENIZER
# =========================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# =========================================================
# FORMAT
# =========================================================

columns = [
    "input_ids",
    "attention_mask",
    "label"
]

train_dataset.set_format(
    type="torch",
    columns=columns
)

validation_dataset.set_format(
    type="torch",
    columns=columns
)

test_dataset.set_format(
    type="torch",
    columns=columns
)


# =========================================================
# MODEL
# =========================================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=-1
    )

    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# =========================================================
# TRAINING ARGUMENTS
# =========================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=2,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    learning_rate=2e-5,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    report_to="none",

    fp16=False
)


# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=DataCollatorWithPadding(
        tokenizer=tokenizer
    ),

    compute_metrics=compute_metrics
)


# =========================================================
# TRAIN
# =========================================================

print("\nStarting Sensitive Content BERT training...")

trainer.train()


# =========================================================
# SAVE
# =========================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print(
    "\nSensitive Content model saved to:"
)

print(
    OUTPUT_DIR
)