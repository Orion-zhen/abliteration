import gc
import torch
from tqdm import tqdm
from transformers import PreTrainedModel


def orthogonalize_vector(
    global_dir: torch.Tensor,
    local_harmless_mean: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Orthogonalizes the global refusal direction with respect to the local harmless mean.
    This implements the "Biprojected" part of the algorithm: protecting the harmless direction of the specific layer being ablated.

    Formula: refusal_local = refusal_global - projection(refusal_global, harmless_local)

    Args:
        global_dir (torch.Tensor): The global refusal direction vector. [hidden_size]
        local_harmless_mean (torch.Tensor): The harmless mean activation value for the current layer. [hidden_size]
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.

    Returns:
        torch.Tensor: The orthogonalized local refusal direction.
    """
    global_dir = global_dir.to(dtype=torch.float32, device=local_harmless_mean.device)
    local_mean = local_harmless_mean.to(dtype=torch.float32)

    # Calculate projection of global_dir onto local_mean
    # proj_u(v) = (v.u) / (u.u) * u
    dot_product = torch.dot(global_dir, local_mean)
    norm_sq = torch.dot(local_mean, local_mean)
    projection = (dot_product / (norm_sq + epsilon)) * local_mean

    # Remove the harmless component
    local_refusal_dir = global_dir - projection

    return local_refusal_dir


def norm_preserving_ablation(
    weight: torch.Tensor,
    refusal_dir: torch.Tensor,
    scale_factor: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.nn.Parameter:
    """Applies Norm-Preserving Abliteration to a weight matrix.

    Logic:

    1. Normalize refusal direction.
    2. Decompose weight matrix into Magnitude (Norm) and Direction.
    3. Calculate projection of weights onto refusal direction.
    4. Ablate the refusal component from the weight direction (Rank-1 update).
    5. Re-normalize the new weight direction (Crucial!).
    6. Recombine original magnitudes with new direction.

    Args:
        weight (torch.Tensor): Original weight matrix. [out_features, in_features]
        refusal_dir (torch.Tensor): Local refusal direction vector. [out_features]
        scale_factor (float, optional): Strength of ablation. Defaults to 1.0.
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.

    Returns:
        torch.nn.Parameter: The modified weight parameter.
    """
    # 1. Setup and fp32 conversion
    device = weight.device
    original_dtype = weight.dtype

    # Move refusal_dir to the same device
    if refusal_dir.device != device:
        refusal_dir = refusal_dir.to(device=device)

    W = weight.data.to(dtype=torch.float32)
    r = refusal_dir.to(dtype=torch.float32)

    # Ensure r is 1D [out_features]
    if r.dim() > 1:
        r = r.view(-1)

    # 2. Normalize refusal direction
    r_normed = torch.nn.functional.normalize(r, dim=0, eps=epsilon)

    # 3. Decompose Weight Matrix
    # Calculate row-wise norms (Magnitudes)
    # W shape: [out, in], Norm shape: [out, 1]
    W_magnitudes = torch.norm(W, dim=1, keepdim=True)
    # Calculate Directional component
    W_direction = torch.nn.functional.normalize(W, dim=1, eps=epsilon)

    # 4. Apply abliteration to the DIRECTIONAL component
    # Calculate projection coefficients: alignment of each raw direction with refusal direction
    # r_normed: [out], W_direction: [out, in] -> projection: [in]
    # Math: p = r^T * W_hat
    projection = torch.matmul(r_normed, W_direction)

    # Remove the refusal component via rank-1 update
    # W_hat_new = W_hat - scale * r_hat * p^T
    # outer(r, p) creates [out, in] matrix
    update_term = torch.outer(r_normed, projection)
    W_direction_new = W_direction - scale_factor * update_term

    # 5. Re-normalize the adjusted direction
    # This ensures that the new direction is still on the unit hyprsphere
    W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1, eps=epsilon)

    # 6. Recombine: keep original magnitude, use new direction
    W_new = W_magnitudes * W_direction_new

    # Cast back to original precesion
    W_final = W_new.to(dtype=original_dtype)

    del (
        W,
        r,
        r_normed,
        W_magnitudes,
        W_direction,
        projection,
        update_term,
        W_direction_new,
        W_new,
    )

    return torch.nn.Parameter(W_final)


def apply_abliteration(
    model: PreTrainedModel,
    global_refusal_dir: torch.Tensor,
    all_harmless_neams: torch.Tensor,
    skip_begin_layers: int = 0,
    skip_end_layers: int = 0,
    scale_factor: float = 1.0,
) -> PreTrainedModel:
    """Iterates through model layers and applies Norm-Preserving Biprojected Abliteration.

    Args:
        model (PreTrainedModel): The LLM model to modify.
        global_refusal_dir (torch.Tensor): The calculated global refusal direction. [hidden_size]
        all_harmless_neams (torch.Tensor): The tensor containing harmless means for all layers. [num_layers, hidden_size]
        skip_begin_layers (int, optional): Number of initial layers to skip. Defaults to 0.
        skip_end_layers (int, optional): Number of final layers to skip. Defaults to 0.
        scale_factor (float, optional): Ablation strength. Defaults to 1.0.

    Returns:
        PreTrainedModel: The modified model.
    """

    lm_model = model.model
    assert hasattr(
        lm_model, "layers"
    ), "The model does not have the expected 'layers' attribute."

    num_layers = len(lm_model.layers)
    target_layers = range(skip_begin_layers, num_layers - skip_end_layers)

    target_layers = range(skip_begin_layers, num_layers - skip_end_layers)

    # Ensure global direction is on CPU initially to avoid OOM
    global_refusal_dir = global_refusal_dir.cpu()
    all_harmless_neams = all_harmless_neams.cpu()

    for layer_idx in tqdm(target_layers, desc="abliterating layers"):
        layer = lm_model.layers[layer_idx]
        # 1. Get Local Harmless Mean for this layer
        if layer_idx < all_harmless_neams.shape[0]:
            local_harmless_mean = all_harmless_neams[layer_idx]
        else:
            # Fallback if dimensions don't match (shouldn't happen with correct logic)
            print(
                f"Warning: No harmless mean found for layer {layer_idx}, using global direction directly."
            )
            local_harmless_mean = torch.zeros_like(global_refusal_dir)

        # 2. Compute Local Refusal Direction
        local_refusal_dir = orthogonalize_vector(
            global_dir=global_refusal_dir, local_harmless_mean=local_harmless_mean
        )

        # 3. Apply to o_proj
        if hasattr(layer.self_attn, "o_proj"):
            layer.self_attn.o_proj.weight = norm_preserving_ablation(
                weight=layer.self_attn.o_proj.weight,
                refusal_dir=local_refusal_dir,
                scale_factor=scale_factor,
            )
        # 4. Apply to down_proj
        if hasattr(layer.mlp, "down_proj"):
            layer.mlp.down_proj.weight = norm_preserving_ablation(
                weight=layer.mlp.down_proj.weight,
                refusal_dir=local_refusal_dir,
                scale_factor=scale_factor,
            )

        # Explicit GC
        if layer_idx % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    return model
