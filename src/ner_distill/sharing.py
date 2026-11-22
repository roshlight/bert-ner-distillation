from pathlib import Path

import torch
from torch import nn
from transformers import (
    BertConfig,
    BertForTokenClassification,
    TrainerCallback,
)

from .models import create_student


def apply_layer_sharing(model, layer_map=(0, 1, 0, 1)):
    layers = model.bert.encoder.layer

    groups = {}
    shared_layers = []

    for position, group in enumerate(layer_map):
        if group not in groups:
            groups[group] = layers[position]
        shared_layers.append(groups[group])

    model.bert.encoder.layer = nn.ModuleList(shared_layers)
    model.config.layer_sharing_map = list(layer_map)

    return model


def create_shared_student(tokenizer):
    student = create_student(tokenizer)
    return apply_layer_sharing(student)


def save_shared_student(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "config": model.config.to_dict(),
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
        },
        path,
    )


def load_shared_student(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    config = BertConfig.from_dict(checkpoint["config"])
    model = BertForTokenClassification(config)

    apply_layer_sharing(
        model,
        layer_map=config.layer_sharing_map,
    )
    model.load_state_dict(checkpoint["state_dict"])

    return model


class SaveBestSharedStudent(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
        self.best_f1 = float("-inf")
        self.best_step = None

    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        model=None,
        **kwargs,
    ):
        score = (metrics or {}).get("eval_f1")

        if score is not None and score > self.best_f1:
            save_shared_student(model, self.path)
            self.best_f1 = float(score)
            self.best_step = state.global_step

            print(
                f"\nСохранён лучший shared student: "
                f"step={self.best_step}, F1={self.best_f1:.4f}"
            )

        return control
    