from transformers import Trainer, TrainingArguments

from .data import LABEL_NAMES
from .metrics import make_compute_metrics


def create_teacher_trainer(
    model,
    dataset,
    tokenized_dataset,
    tokenizer,
    data_collator,
    output_dir="/kaggle/working/outputs/teacher",
):
    model.config.id2label = dict(enumerate(LABEL_NAMES))
    model.config.label2id = {
        name: index for index, name in enumerate(LABEL_NAMES)
    }

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        warmup_steps=260,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
        seed=42,
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(
            dataset["validation"], tokenizer
        ),
    )
