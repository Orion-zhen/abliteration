import torch
from typing import Optional, Literal


def magnitude_clip(tensor: torch.Tensor, max_val: float) -> torch.Tensor:
    """Clips tensor values to [-max_val, max_val] by magnitude preserving sign."""
    return torch.clamp(tensor, -max_val, max_val)


def remove_orthogonal_projection(
    refusal_direction: torch.Tensor, orthogonal_base: torch.Tensor
) -> torch.Tensor:
    """
    Removes the component of refusal_direction containing the projection onto orthogonal_base.

    formula: v = v - dot(v, u) * u, where u is the normalized orthogonal_base.

    Args:
        refusal_direction (torch.Tensor): The vector to modify.
        orthogonal_base (torch.Tensor): The basis vector to project onto.

    Returns:
        torch.Tensor: The modified refusal_direction. result is NOT normalized.
    """
    # Ensure base is normalized for projection
    # Using dim=0 as these are expected to be 1D vectors [hidden_size]
    # or compatible shapes where dim=0 is the reduction dimension if they are flattened.
    orthogonal_base_normed = torch.nn.functional.normalize(orthogonal_base, dim=0)

    # Calculate projection scalar: <v, u>
    projection = torch.dot(refusal_direction, orthogonal_base_normed)

    # Remove projection: v - <v, u> * u
    result = refusal_direction - projection * orthogonal_base_normed

    return result


def modify_tensor_simple(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.Tensor:
    """
    Modify tensor by calculating the outer product of the refusal direction and the refusal direction.
    Equation: W_new = W - scale * (r @ r.T) @ W
    (Assuming W is [out, in] and r is [out])

    This does NOT preserve the norm of the weights.
    """
    # Force CPU
    original_dtype = tensor_data.dtype
    tensor_data = tensor_data.to("cpu")
    device = tensor_data.device  # Should be cpu now

    if refusal_dir.device != device:
        refusal_dir = refusal_dir.to(device)

    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32)

    # Ensure refusal_dir is a 1-dimensional tensor
    if refusal_dir_float32.dim() > 1:
        refusal_dir_float32 = refusal_dir_float32.view(-1)

    update = torch.matmul(
        torch.outer(refusal_dir_float32, refusal_dir_float32), tensor_float32
    )

    tensor_modified = tensor_float32 - (scale_factor * update)

    result = tensor_modified.to(original_dtype)

    # Cleanup
    del tensor_float32, refusal_dir_float32, update, tensor_modified
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def modify_tensor_norm_preserved(
    W: torch.Tensor,
    refusal_dir: torch.Tensor,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating refusal direction while preserving row norms.
    Strictly follows reference implementation logic regarding Safetensors transposition.

    Args:
        W: Weight tensor (Safetensors format: [in_features, out_features])
        refusal_dir: Refusal direction vector [out_features]
        scale_factor: Scale factor for ablation
    """
    original_dtype = W.dtype
    device = "cpu"  # Force CPU

    with torch.no_grad():
        # Move tensors for computation
        # Transpose here to convert from safetensors convention [in, out] -> [out, in]
        # This aligns with PyTorch's nn.Linear convention for calculation
        W_cpu = W.to(device=device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_cpu = refusal_dir.to(
            device=device, dtype=torch.float32, non_blocking=True
        )

        # Ensure refusal_dir is a 1-dimensional tensor
        if refusal_dir_cpu.dim() > 1:
            refusal_dir_cpu = refusal_dir_cpu.view(-1)

        # Normalize refusal direction
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_cpu, dim=0)

        # Decompose weight matrix
        # W_cpu is [out_features, in_features]
        # Calculate row norms (magnitude of each output neuron's weight vector)
        W_norm: torch.Tensor = torch.norm(
            W_cpu, dim=1, keepdim=True
        )  # [out_features, 1]

        # Normalize weights to get direction
        W_direction = torch.nn.functional.normalize(W_cpu, dim=1)

        # Apply abliteration to the DIRECTIONAL component
        # Compute dot product of each row with refusal direction: p = d . r
        projection = torch.matmul(W_direction, refusal_normalized)  # [in_features]

        # Subtract the projection from the direction: d_new = d - scale * p * r
        W_direction_new = W_direction - scale_factor * torch.outer(
            projection, refusal_normalized
        )

        # Re-normalize the adjusted direction to ensure it stays on the unit hypersphere
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)

        # Recombine: keep original magnitude, use new direction
        W_modified = W_norm * W_direction_new

        # Convert back to original dtype and CPU
        # Transpose here to return to safetensors convention [out, in] -> [in, out]
        result = W_modified.T.to(device=device, dtype=original_dtype, non_blocking=True)

        # Cleanup
        del W_cpu, refusal_dir_cpu, refusal_normalized, projection
        del W_direction, W_direction_new, W_norm, W_modified

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return result.detach().clone()


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


def soft_threshold_sparsify(
    vector: torch.Tensor, threshold: float = 0.01
) -> torch.Tensor:
    """
    Apply soft thresholding (L1 regularization-style sparsification).
    """
    return torch.sign(vector) * torch.clamp(torch.abs(vector) - threshold, min=0)


def sparsify_tensor(
    tensor: torch.Tensor,
    method: Literal["magnitude", "percentile", "topk", "soft_threshold"] = "magnitude",
    threshold: Optional[float] = None,
    **kwargs,
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
