import numpy as np
from seqeval.metrics import f1_score

from .data import LABEL_NAMES


def make_compute_metrics(raw_split, tokenizer, max_length=256):
    encoded = tokenizer(
        raw_split["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    first_token_positions = []

    for index in range(len(raw_split)):
        positions = []
        previous_word_id = None

        for position, word_id in enumerate(encoded.word_ids(index)):
            if word_id is not None and word_id != previous_word_id:
                positions.append(position)
            previous_word_id = word_id

        first_token_positions.append(positions)

    def compute_metrics(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        predictions = np.argmax(logits, axis=-1)
        labels = eval_prediction.label_ids

        predicted_tags = [
            [LABEL_NAMES[int(predictions[i, pos])] for pos in positions]
            for i, positions in enumerate(first_token_positions)
        ]
        true_tags = [
            [LABEL_NAMES[int(labels[i, pos])] for pos in positions]
            for i, positions in enumerate(first_token_positions)
        ]

        return {"f1": f1_score(true_tags, predicted_tags)}

    return compute_metrics
