import torch
import gc

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        # Move tensors for computation
        # Transpose here to convert from safetensors convention [in, out] -> [out, in]
        # This aligns with PyTorch's nn.Linear convention for calculation
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure refusal_dir is a 1-dimensional tensor
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        
        # Normalize refusal direction
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)

        # Decompose weight matrix
        # W_gpu is [out_features, in_features]
        # Calculate row norms (magnitude of each output neuron's weight vector)
        W_norm = torch.norm(W_gpu, dim=1, keepdim=True)  # [out_features, 1]
        
        # Normalize weights to get direction
        W_direction = torch.nn.functional.normalize(W_gpu, dim=1)
    
        # Apply abliteration to the DIRECTIONAL component
        # Compute dot product of each row with refusal direction: p = d . r
        projection = torch.matmul(W_direction, refusal_normalized)  # [in_features]
        
        # Subtract the projection from the direction: d_new = d - scale * p * r
        W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
    
        # Re-normalize the adjusted direction to ensure it stays on the unit hypersphere
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
    
        # Recombine: keep original magnitude, use new direction
        W_modified = W_norm * W_direction_new
        
        # Convert back to original dtype and CPU
        # Transpose here to return to safetensors convention [out, in] -> [in, out]
        result = W_modified.T.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup
        del W_gpu, refusal_dir_gpu, refusal_normalized, projection
        del W_direction, W_direction_new, W_norm, W_modified
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return result.detach().clone()
