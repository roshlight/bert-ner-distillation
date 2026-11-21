import torch
from torch import nn


class FactorizedEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        rank=64,
        padding_idx=None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings,
            rank,
            padding_idx=padding_idx,
        )
        self.projection = nn.Linear(
            rank,
            embedding_dim,
            bias=False,
        )

    def forward(self, input_ids):
        return self.projection(self.embedding(input_ids))

    @classmethod
    def from_embedding(cls, original, rank=64):
        weight = original.weight.detach().to(
            device="cpu", dtype=torch.float32
        )

        if not 1 <= rank <= min(weight.shape):
            raise ValueError("Недопустимый ранг факторизации")

        u, s, vh = torch.linalg.svd(weight, full_matrices=False)

        result = cls(
            num_embeddings=original.num_embeddings,
            embedding_dim=original.embedding_dim,
            rank=rank,
            padding_idx=original.padding_idx,
        )

        sqrt_s = s[:rank].sqrt()

        with torch.no_grad():
            result.embedding.weight.copy_(
                u[:, :rank] * sqrt_s
            )
            result.projection.weight.copy_(
                (sqrt_s[:, None] * vh[:rank, :]).T
            )

        return result.to(
            device=original.weight.device,
            dtype=original.weight.dtype,
        )


def factorize_word_embeddings(model, rank=64):
    original = model.bert.embeddings.word_embeddings

    model.bert.embeddings.word_embeddings = (
        FactorizedEmbedding.from_embedding(original, rank)
    )

    return model
