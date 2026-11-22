import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer, TrainingArguments

from .metrics import make_compute_metrics


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher = teacher.to(self.args.device)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"]
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "labels"
        }

        student_outputs = model(**model_inputs)
        student_logits = student_outputs.logits

        mask = labels != -100
        if "attention_mask" in model_inputs:
            mask = mask & model_inputs["attention_mask"].bool()

        student_valid = student_logits[mask].float()
        labels_valid = labels[mask]

        hard_loss = F.cross_entropy(
            student_valid,
            labels_valid,
        )

        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(**model_inputs).logits
            teacher_valid = teacher_logits[mask].float()

        temperature = self.temperature

        soft_loss = self.kl_loss(
            F.log_softmax(student_valid / temperature, dim=-1),
            F.softmax(teacher_valid / temperature, dim=-1),
        ) * temperature**2

        loss = hard_loss + soft_loss

        if return_outputs:
            return loss, student_outputs
        return loss


def create_distillation_trainer(
    student,
    teacher,
    dataset,
    tokenized_dataset,
    tokenizer,
    data_collator,
    max_steps=20_000,
):
    args = TrainingArguments(
        output_dir="/kaggle/working/outputs/student",
        max_steps=max_steps,
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        warmup_steps=int(0.1 * max_steps),
        lr_scheduler_type="linear",
        fp16=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    return DistillationTrainer(
        model=student,
        teacher=teacher,
        temperature=2.0,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(
            dataset["validation"], tokenizer
        ),
    )
