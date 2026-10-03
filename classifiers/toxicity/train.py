import os
import numpy as np
import torch

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from preprocessing import (
    load_and_clean_data,
    LABELS
)

from dataset import ToxicDataset


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "bert-base-uncased"

DATA_PATH = "../data/train.csv"

MODEL_PATH = "../models/bert-toxic-model-v3"

RESULTS_PATH = "../models/evaluation-v3"

MAX_LENGTH = 64

# Keep this at 5000 for the V3 experiment
DATASET_SIZE = 20000


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)


# =========================================================
# LOAD DATA
# =========================================================

df = load_and_clean_data(DATA_PATH)

print(
    "\nOriginal dataset size:",
    len(df)
)


if DATASET_SIZE is not None:

    df = df.sample(
        DATASET_SIZE,
        random_state=42
    ).reset_index(drop=True)

print(
    "Experiment dataset size:",
    len(df)
)


# =========================================================
# DATA
# =========================================================

X = df["clean_text"].values
Y = df[LABELS].values


# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print("\nLabel distribution:")

for i, label in enumerate(LABELS):

    print(
        f"{label:<15}: "
        f"{Y[:, i].sum()}"
    )


# =========================================================
# STRATIFIED TRAIN / TEMP SPLIT
# =========================================================

splitter = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(X, Y)
)


X_train = X[train_idx]
Y_train = Y[train_idx]

X_temp = X[temp_idx]
Y_temp = Y[temp_idx]


# =========================================================
# STRATIFIED VALIDATION / TEST SPLIT
# =========================================================

splitter2 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    splitter2.split(X_temp, Y_temp)
)


X_val = X_temp[val_idx]
Y_val = Y_temp[val_idx]

X_test = X_temp[test_idx]
Y_test = Y_temp[test_idx]


print("\nDataset split:")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# =========================================================
# TOKENIZER
# =========================================================

print("\nLoading tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# TOKENIZATION
# =========================================================

print("Tokenizing training data...")

train_encodings = tokenizer(
    X_train.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)


print("Tokenizing validation data...")

val_encodings = tokenizer(
    X_val.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)


print("Tokenizing test data...")

test_encodings = tokenizer(
    X_test.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)


# =========================================================
# DATASETS
# =========================================================

train_dataset = ToxicDataset(
    train_encodings,
    Y_train
)

val_dataset = ToxicDataset(
    val_encodings,
    Y_val
)

test_dataset = ToxicDataset(
    test_encodings,
    Y_test
)


# =========================================================
# CLASS WEIGHTS
# =========================================================

positive_counts = Y_train.sum(axis=0)

negative_counts = len(Y_train) - positive_counts

# Positive class weight:
# negatives / positives

pos_weights = (
    negative_counts /
    np.maximum(positive_counts, 1)
)

# Prevent extremely large weights
pos_weights = np.clip(
    pos_weights,
    1.0,
    10.0
)

pos_weights_tensor = torch.tensor(
    pos_weights,
    dtype=torch.float
)

print("\nClass weights:")

for label, weight in zip(
    LABELS,
    pos_weights
):

    print(
        f"{label:<15}: {weight:.2f}"
    )


# =========================================================
# MODEL
# =========================================================

print("\nLoading BERT model...")

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
)

model.to(device)


# =========================================================
# CUSTOM TRAINER
# =========================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        weights = pos_weights_tensor.to(
            logits.device
        )

        loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=weights
        )

        loss = loss_function(
            logits,
            labels
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    probabilities = torch.sigmoid(
        torch.tensor(logits)
    )

    predictions = (
        probabilities >= 0.5
    ).int().numpy()

    return {
        "f1_micro": f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    }


# =========================================================
# TRAINING ARGUMENTS
# =========================================================

training_args = TrainingArguments(

    output_dir="../models/checkpoints-v3",

    num_train_epochs=1,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    gradient_accumulation_steps=4,

    learning_rate=2e-5,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="f1_macro",

    greater_is_better=True,

    save_total_limit=1,

    report_to="none",

    fp16=torch.cuda.is_available(),

    seed=42
)


# =========================================================
# TRAINER
# =========================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    compute_metrics=compute_metrics
)


# =========================================================
# TRAIN
# =========================================================

print("\n========================================")
print("Starting BERT V3 training")
print("Class-weighted loss enabled")
print("========================================\n")

trainer.train()


# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

print("\nGenerating validation predictions...")

val_output = trainer.predict(
    val_dataset
)

val_logits = val_output.predictions

val_probabilities = torch.sigmoid(
    torch.tensor(val_logits)
).numpy()


# =========================================================
# FIND THRESHOLDS USING VALIDATION ONLY
# =========================================================

print("\n========================================")
print("VALIDATION THRESHOLD OPTIMIZATION")
print("========================================")

best_thresholds = {}

for i, label in enumerate(LABELS):

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(
        0.05,
        0.91,
        0.05
    ):

        predictions = (
            val_probabilities[:, i]
            >= threshold
        ).astype(int)

        score = f1_score(
            Y_val[:, i],
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    best_thresholds[label] = best_threshold

    print(
        f"{label:<15} "
        f"Threshold: {best_threshold:.2f} "
        f"Validation F1: {best_f1:.4f}"
    )


# =========================================================
# TEST PREDICTIONS
# =========================================================

print("\nGenerating final test predictions...")

test_output = trainer.predict(
    test_dataset
)

test_logits = test_output.predictions

test_probabilities = torch.sigmoid(
    torch.tensor(test_logits)
).numpy()


# =========================================================
# APPLY VALIDATION THRESHOLDS TO TEST
# =========================================================

test_predictions = np.zeros_like(
    test_probabilities,
    dtype=int
)

for i, label in enumerate(LABELS):

    threshold = best_thresholds[label]

    test_predictions[:, i] = (
        test_probabilities[:, i]
        >= threshold
    ).astype(int)


# =========================================================
# FINAL TEST RESULTS
# =========================================================

print("\n========================================")
print("FINAL TEST RESULTS")
print("Validation thresholds applied")
print("========================================")


for i, label in enumerate(LABELS):

    precision = precision_score(
        Y_test[:, i],
        test_predictions[:, i],
        zero_division=0
    )

    recall = recall_score(
        Y_test[:, i],
        test_predictions[:, i],
        zero_division=0
    )

    f1 = f1_score(
        Y_test[:, i],
        test_predictions[:, i],
        zero_division=0
    )

    print(f"\n{label}")

    print(
        f"  Precision: {precision:.4f}"
    )

    print(
        f"  Recall:    {recall:.4f}"
    )

    print(
        f"  F1:        {f1:.4f}"
    )


# =========================================================
# OVERALL METRICS
# =========================================================

micro_f1 = f1_score(
    Y_test,
    test_predictions,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    Y_test,
    test_predictions,
    average="macro",
    zero_division=0
)


print("\n========================================")
print("OVERALL TEST PERFORMANCE")
print("========================================")

print(
    f"Micro F1: {micro_f1:.4f}"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)


# =========================================================
# SAVE RESULTS
# =========================================================

os.makedirs(
    RESULTS_PATH,
    exist_ok=True
)

np.savez(
    f"{RESULTS_PATH}/results.npz",
    probabilities=test_probabilities,
    predictions=test_predictions,
    labels=Y_test
)

np.save(
    f"{RESULTS_PATH}/best_thresholds.npy",
    best_thresholds
)


# =========================================================
# SAVE MODEL
# =========================================================

model.save_pretrained(
    MODEL_PATH
)

tokenizer.save_pretrained(
    MODEL_PATH
)


print(
    "\nModel saved successfully at:"
)

print(MODEL_PATH)

print(
    "\nV3 training and evaluation complete."
)