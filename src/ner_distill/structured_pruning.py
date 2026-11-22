import torch
from torch import nn
from transformers import BertConfig, BertForTokenClassification

from .factorization import FactorizedEmbedding


class PrunedStudent(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)

        self.bert.embeddings.word_embeddings = FactorizedEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            rank=config.embedding_rank,
            padding_idx=config.pad_token_id,
        )


@torch.no_grad()
def select_ffn_neurons(
    teacher,
    calibration_loader,
    layer_indices=(2, 5, 8, 11),
    intermediate_size=1024,
    device="cuda",
):
    teacher.to(device).eval()

    scores = {
        index: torch.zeros(
            teacher.config.intermediate_size,
            device=device,
            dtype=torch.float32,
        )
        for index in layer_indices
    }

    current_mask = None

    def make_hook(layer_index):
        def hook(module, inputs):
            # Вход output.dense — активации после intermediate GELU.
            activations = inputs[0].detach()
            valid = activations[current_mask].float()

            scores[layer_index].add_(valid.abs().sum(dim=0))

        return hook

    handles = [
        teacher.bert.encoder.layer[index].output.dense
        .register_forward_pre_hook(make_hook(index))
        for index in layer_indices
    ]

    try:
        for batch in calibration_loader:
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            current_mask = model_inputs["attention_mask"].bool()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                teacher(**model_inputs)
    finally:
        for handle in handles:
            handle.remove()

    return {
        index: torch.topk(
            scores[index],
            k=intermediate_size,
        ).indices.sort().values.cpu()
        for index in layer_indices
    }


@torch.no_grad()
def build_pruned_student(
    teacher,
    selected_neurons,
    layer_indices=(2, 5, 8, 11),
    intermediate_size=1024,
    embedding_rank=64,
):
    teacher.cpu().float()

    config = BertConfig.from_dict(teacher.config.to_dict())
    config.num_hidden_layers = len(layer_indices)
    config.intermediate_size = intermediate_size
    config.embedding_rank = embedding_rank
    config.source_teacher_layers = list(layer_indices)
    config.selected_ffn_neurons = [
        selected_neurons[index].tolist()
        for index in layer_indices
    ]

    student = PrunedStudent(config)

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()

    for target_name, target_tensor in student_state.items():
        # Эти веса ниже получим через SVD.
        if target_name.startswith("bert.embeddings.word_embeddings."):
            continue

        source_name = target_name

        if target_name.startswith("bert.encoder.layer."):
            student_index = int(target_name.split(".")[3])
            teacher_index = layer_indices[student_index]

            source_name = target_name.replace(
                f"bert.encoder.layer.{student_index}.",
                f"bert.encoder.layer.{teacher_index}.",
                1,
            )

            source_tensor = teacher_state[source_name]
            neurons = selected_neurons[teacher_index]

            if target_name.endswith("intermediate.dense.weight"):
                source_tensor = source_tensor[neurons, :]

            elif target_name.endswith("intermediate.dense.bias"):
                source_tensor = source_tensor[neurons]

            elif target_name.endswith("output.dense.weight"):
                if ".attention." not in target_name:
                    source_tensor = source_tensor[:, neurons]

        else:
            source_tensor = teacher_state[source_name]

        target_tensor.copy_(source_tensor)

    student.bert.embeddings.word_embeddings = (
        FactorizedEmbedding.from_embedding(
            teacher.bert.embeddings.word_embeddings,
            rank=embedding_rank,
        )
    )

    return student
