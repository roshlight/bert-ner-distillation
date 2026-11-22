import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    BertConfig,
    BertForTokenClassification,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput

from .data import LABEL_NAMES
from .distillation import DistillationTrainer
from .metrics import make_compute_metrics


class HiddenStudent(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)

        self.hidden_projections = nn.ModuleList([
            nn.Linear(
                config.hidden_size,
                config.teacher_hidden_size,
                bias=False,
            )
            for _ in config.teacher_layer_indices
        ])

        self.hidden_projections.apply(self._init_weights)


def create_hidden_student(tokenizer, teacher_hidden_size=768):
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

    config.teacher_hidden_size = teacher_hidden_size
    config.teacher_layer_indices = [3, 6, 9, 12]
    config.hidden_loss_weight = 1.0

    return HiddenStudent(config)


class HiddenDistillationTrainer(DistillationTrainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        student_outputs = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        self.teacher.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        mask = labels != -100
        if "attention_mask" in model_inputs:
            mask = mask & model_inputs["attention_mask"].bool()

        student_logits = student_outputs.logits[mask].float()
        teacher_logits = teacher_outputs.logits[mask].float()

        hard_loss = F.cross_entropy(student_logits, labels[mask])
        temperature = self.temperature

        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
        ) * temperature**2

        student_model = self.accelerator.unwrap_model(model)

        hidden_losses = []

        for student_index, teacher_index in enumerate(
            student_model.config.teacher_layer_indices,
            start=1,
        ):
            student_hidden = student_outputs.hidden_states[student_index][mask]
            teacher_hidden = teacher_outputs.hidden_states[teacher_index][mask]
            projected = student_model.hidden_projections[student_index - 1](student_hidden)

            hidden_losses.append(F.mse_loss(projected.float(), teacher_hidden.float()))

        hidden_loss = torch.stack(hidden_losses).mean()

        loss = (
            hard_loss
            + soft_loss
            + student_model.config.hidden_loss_weight * hidden_loss
        )

        if return_outputs:
            outputs = TokenClassifierOutput(
                loss=loss,
                logits=student_outputs.logits,
            )
            return loss, outputs

        return loss


def create_hidden_trainer(
    student,
    teacher,
    dataset,
    tokenized_dataset,
    tokenizer,
    data_collator,
    output_dir,
    max_steps=5000,
):
    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        warmup_steps=int(0.1 * max_steps),
        lr_scheduler_type="linear",
        fp16=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    return HiddenDistillationTrainer(
        model=student,
        teacher=teacher,
        temperature=2.0,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(
            dataset["validation"],
            tokenizer,
        ),
    )
