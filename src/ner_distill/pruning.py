import torch
from torch import nn


@torch.no_grad()
def collect_hessians(model, modules, calibration_loader, device):
    statistics = {
        name: {
            "gram": torch.zeros(
                layer.in_features,
                layer.in_features,
                device=device,
                dtype=torch.float32,
            ),
            "count": 0,
        }
        for name, layer in modules.items()
    }

    current_mask = None

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0].detach().reshape(-1, module.in_features)
            x = x[current_mask.reshape(-1)].float()

            statistics[name]["gram"].add_(x.T @ x)
            statistics[name]["count"] += x.shape[0]

        return hook

    handles = [
        layer.register_forward_hook(make_hook(name))
        for name, layer in modules.items()
    ]

    try:
        for batch in calibration_loader:
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            current_mask = model_inputs["attention_mask"].bool()
            model(**model_inputs)
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: 2.0 * values["gram"] / values["count"]
        for name, values in statistics.items()
    }


@torch.no_grad()
def prune_wanda(linear, hessian, sparsity=0.5):
    weight = linear.weight.detach().float().clone()

    input_scale = hessian.diagonal().clamp_min(0).sqrt()
    importance = weight.abs() * input_scale.unsqueeze(0)

    prune_count = int(sparsity * weight.shape[1])
    indices = importance.argsort(dim=1)[:, :prune_count]

    weight.scatter_(1, indices, 0)
    linear.weight.copy_(weight.to(linear.weight.dtype))


@torch.no_grad()
def prune_sparsegpt(
    linear,
    hessian,
    sparsity=0.5,
    block_size=128,
    damping=0.01,
):
    weight = linear.weight.detach().float().clone()
    hessian = hessian.float().clone()

    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1.0
    weight[:, dead] = 0.0

    hessian = (hessian + hessian.T) * 0.5

    diagonal = torch.arange(hessian.shape[0], device=hessian.device)
    damp_value = damping * hessian.diagonal().mean()
    hessian[diagonal, diagonal] += damp_value

    chol = torch.linalg.cholesky(hessian)
    inverse = torch.cholesky_inverse(chol)
    inverse = (inverse + inverse.T) * 0.5

    upper = torch.linalg.cholesky(inverse, upper=True)

    column_count = weight.shape[1]

    for start in range(0, column_count, block_size):
        end = min(start + block_size, column_count)

        block = weight[:, start:end].clone()
        upper_block = upper[start:end, start:end]

        pruned_block = torch.zeros_like(block)
        errors = torch.zeros_like(block)

        importance = (
            block.square()
            / upper_block.diagonal().square().unsqueeze(0)
        )

        prune_count = int(sparsity * block.numel())

        flat_mask = torch.zeros(
            block.numel(),
            dtype=torch.bool,
            device=block.device,
        )
        indices = torch.topk(
            importance.flatten(),
            k=prune_count,
            largest=False,
        ).indices
        flat_mask[indices] = True
        mask = flat_mask.reshape_as(block)

        for column in range(end - start):
            original_column = block[:, column].clone()
            pruned_column = original_column.masked_fill(
                mask[:, column], 0.0
            )

            pruned_block[:, column] = pruned_column

            error = (
                original_column - pruned_column
            ) / upper_block[column, column]

            errors[:, column] = error

            block[:, column:] -= (
                error.unsqueeze(1)
                * upper_block[column, column:].unsqueeze(0)
            )

        weight[:, start:end] = pruned_block

        weight[:, end:] -= errors @ upper[start:end, end:]

    linear.weight.copy_(weight.to(linear.weight.dtype))


@torch.no_grad()
def prune_bert(
    model,
    calibration_loader,
    method="sparsegpt",
    sparsity=0.5,
    block_size=128,
    damping=0.01,
    device="cuda",
):
    model.to(device).float().eval()

    groups = [
        [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
        ],
        ["attention.output.dense"],
        ["intermediate.dense"],
        ["output.dense"],
    ]

    for layer_index, layer in enumerate(model.bert.encoder.layer):
        for names in groups:
            modules = {
                name: layer.get_submodule(name)
                for name in names
            }

            hessians = collect_hessians(
                model,
                modules,
                calibration_loader,
                device,
            )

            for name, linear in modules.items():
                if method == "sparsegpt":
                    prune_sparsegpt(
                        linear,
                        hessians[name],
                        sparsity=sparsity,
                        block_size=block_size,
                        damping=damping,
                    )
                elif method == "wanda":
                    prune_wanda(
                        linear,
                        hessians[name],
                        sparsity=sparsity,
                    )
                else:
                    raise ValueError(f"Неизвестный метод: {method}")

            del hessians

        print(
            f"{method}: обработан слой "
            f"{layer_index + 1}/{len(model.bert.encoder.layer)}"
        )

    return model


@torch.no_grad()
def sparsity_statistics(model):
    parameters = list(model.parameters())

    encoder_weights = [
        module.weight
        for module in model.bert.encoder.modules()
        if isinstance(module, nn.Linear)
    ]

    total = sum(parameter.numel() for parameter in parameters)
    nonzero = sum(
        torch.count_nonzero(parameter).item()
        for parameter in parameters
    )

    encoder_total = sum(weight.numel() for weight in encoder_weights)
    encoder_nonzero = sum(
        torch.count_nonzero(weight).item()
        for weight in encoder_weights
    )

    return {
        "total_parameters": total,
        "nonzero_parameters": nonzero,
        "whole_model_sparsity": 1.0 - nonzero / total,
        "encoder_linear_sparsity": (
            1.0 - encoder_nonzero / encoder_total
        ),
    }
