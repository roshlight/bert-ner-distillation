from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification


LABEL_NAMES = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC",
]

LABEL_TO_ID = {name: index for index, name in enumerate(LABEL_NAMES)}


def align_labels_with_tokens(labels, tokenized_text, batch_index=0):
    aligned_labels = []
    previous_word_id = None

    for word_id in tokenized_text.word_ids(batch_index=batch_index):
        if word_id is None:
            label_id = -100
        else:
            label_id = labels[word_id]

            if word_id == previous_word_id:
                label_name = LABEL_NAMES[label_id]
                if label_name.startswith("B-"):
                    label_id = LABEL_TO_ID["I-" + label_name[2:]]

        aligned_labels.append(label_id)
        previous_word_id = word_id

    return aligned_labels


def load_ner_data(max_length=256):
    dataset = load_dataset("ashaba1in/conll2003")

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased",
        use_fast=True,
    )

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        encoded["labels"] = [
            align_labels_with_tokens(
                labels,
                encoded,
                batch_index=index,
            )
            for index, labels in enumerate(batch["ner_tags"])
        ]
        return encoded

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Токенизация и выравнивание BIO",
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    return dataset, tokenized_dataset, tokenizer, data_collator
