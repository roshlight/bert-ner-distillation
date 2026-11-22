from transformers import BertConfig, BertForTokenClassification

from .data import LABEL_NAMES


def create_student(tokenizer):
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=len(LABEL_NAMES),
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id={
            name: index for index, name in enumerate(LABEL_NAMES)
        },
    )

    student = BertForTokenClassification(config)
    return student
