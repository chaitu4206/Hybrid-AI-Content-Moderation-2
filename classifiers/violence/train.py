import os

from datasets import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments
)

from preprocessing import (
    preprocess_function,
    id2label,
    label2id
)

from dataset import load_dataset


MODEL_NAME = "bert-base-uncased"

OUTPUT_DIR = "models/violence_bert"

MAX_LENGTH = 64


def main():

    print("=" * 60)
    print("VIOLENCE BERT CLASSIFIER TRAINING")
    print("=" * 60)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    train_df, val_df, test_df = load_dataset()

    # --------------------------------------------------
    # CONVERT TO HF DATASETS
    # --------------------------------------------------

    train_dataset = Dataset.from_pandas(
        train_df
    )

    val_dataset = Dataset.from_pandas(
        val_df
    )

    test_dataset = Dataset.from_pandas(
        test_df
    )

    # --------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------

    print("\nLoading BERT tokenizer...")

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize_function(examples):

        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    print("\nTokenizing datasets...")

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

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    print("\nLoading BERT model...")

    model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "NOT_VIOLENCE",
        1: "VIOLENCE"
    },
    label2id={
        "NOT_VIOLENCE": 0,
        "VIOLENCE": 1
    }
)

    # --------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="steps",

        logging_steps=50,

        learning_rate=2e-5,

        per_device_train_batch_size=8,

        per_device_eval_batch_size=8,

        num_train_epochs=2,

        weight_decay=0.01,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        report_to="none",

        save_total_limit=1
    )

    # --------------------------------------------------
    # TRAINER
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        processing_class=tokenizer
    )

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    print("\nStarting training...\n")

    trainer.train()

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    print("\nSaving final model...")

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

    print("\nTraining completed!")

    print(
        f"Model saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()