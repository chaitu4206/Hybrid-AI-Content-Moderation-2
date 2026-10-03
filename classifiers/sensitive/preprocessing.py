from transformers import AutoTokenizer


MODEL_NAME = "bert-base-uncased"

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


def tokenize_dataset(dataset):

    return dataset.map(
        tokenize_function,
        batched=True
    )