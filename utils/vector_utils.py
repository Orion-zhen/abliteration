import torch

def remove_orthogonal_projection(refusal_direction: torch.Tensor, orthogonal_base: torch.Tensor) -> torch.Tensor:
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
    # In the context of abliteration, these are typically 1D vectors.
    orthogonal_base_normed = torch.nn.functional.normalize(orthogonal_base, dim=0)
    
    # Calculate projection scalar: <v, u>
    projection = torch.dot(refusal_direction, orthogonal_base_normed)
    
    # Remove projection: v - <v, u> * u
    result = refusal_direction - projection * orthogonal_base_normed
    
    return result
