import torch
from torch import nn
from torchtyping import TensorType

class CLIPRenderer(nn.Module):
    """Calculate CLIP embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: TensorType["bs":..., "num_samples", "num_classes"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        """Calculate semantics along the ray."""
        # 4096 24 512
        # 4096 24 1
        output = torch.sum(weights * embeds, dim=-2)
        # 4096 512
        output = output / torch.linalg.norm(output, dim=-1, keepdim=True)
        return output


class MeanRenderer(nn.Module):
    """Calculate average of embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: TensorType["bs":..., "num_samples", "num_classes"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        return output