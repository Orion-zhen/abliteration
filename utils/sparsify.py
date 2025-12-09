import torch
from typing import Optional, Literal, Union

def magnitude_sparsify(vector: torch.Tensor, fraction: float = 0.05) -> torch.Tensor:
    """
    Keep components with magnitude >= fraction * max(|vector|).
    """
    threshold = fraction * vector.abs().max()
    return torch.where(vector.abs() >= threshold, vector, torch.zeros_like(vector))


def percentile_sparsify(vector: torch.Tensor, percentile: float = 0.95) -> torch.Tensor:
    """
    Keep components above the given percentile by magnitude.
    """
    threshold = torch.quantile(vector.abs().float(), percentile)
    return torch.where(vector.abs() >= threshold, vector, torch.zeros_like(vector))


def topk_sparsify(vector: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only the top k components by magnitude.
    """
    flat_vector = vector.view(-1)
    _, indices = torch.topk(torch.abs(flat_vector), min(k, flat_vector.numel()))
    mask = torch.zeros_like(flat_vector, dtype=torch.bool)
    mask[indices] = True
    return vector * mask.view(vector.shape)


def soft_threshold_sparsify(vector: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    Apply soft thresholding (L1 regularization-style sparsification).
    """
    return torch.sign(vector) * torch.clamp(torch.abs(vector) - threshold, min=0)


def sparsify_tensor(
    tensor: torch.Tensor,
    method: Literal["magnitude", "percentile", "topk", "soft_threshold"] = "magnitude",
    threshold: Optional[float] = None,
    **kwargs
) -> torch.Tensor:
    """
    Generic entry point for sparsification.
    """
    if method == "magnitude":
        return magnitude_sparsify(tensor, threshold or 0.05)
    elif method == "percentile":
        return percentile_sparsify(tensor, threshold or 0.95)
    elif method == "topk":
        k = kwargs.get("k", int(0.1 * tensor.numel()))
        return topk_sparsify(tensor, k)
    elif method == "soft_threshold":
        return soft_threshold_sparsify(tensor, threshold or 0.01)
    else:
        raise ValueError(f"Unknown method: {method}")

def sparsity_stats(tensor: torch.Tensor) -> dict:
    """
    Compute sparsification statistics for a tensor.
    """
    total_components = tensor.numel()
    nonzero_components = torch.count_nonzero(tensor).item()
    sparsity = 1.0 - (nonzero_components / total_components)
    
    abs_tensor = tensor.abs()
    
    return {
        "total_components": total_components,
        "nonzero_components": nonzero_components,
        "sparsity": sparsity,
        "max_magnitude": abs_tensor.max().item(),
        "mean_magnitude": abs_tensor.mean().item(),
    }
