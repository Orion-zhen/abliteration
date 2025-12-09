
import gc
import torch
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
from safetensors.torch import load_file, save_file

from utils.config import ModelConfig
from utils.io import resolve_model_paths, copy_model_artifacts
from utils.math_utils import (
    modify_tensor_norm_preserved, 
    modify_tensor_simple, 
    sparsify_tensor,
    remove_orthogonal_projection
)

def get_layer_ablation_config(
    config: ModelConfig,
    layer_idx: int,
    global_refusal_dir: torch.Tensor,
    measurement_results: Dict[str, torch.Tensor]
) -> Tuple[float, torch.Tensor, bool]:
    """
    Determines the ablation configuration for a specific layer.
    Returns: scale, refusal_dir, use_norm_preserving
    """
    scale = config.ablation.global_scale
    refusal_dir = global_refusal_dir
    
    # Check Overrides
    override = config.ablation.layer_overrides.get(layer_idx)
    if override is None:
         override = config.ablation.layer_overrides.get(str(layer_idx)) # Retry if the key is string

    if override:
        scale = override.get("scale", scale)
        source_idx = override.get("source_layer")
        if source_idx is not None:
             # Use specific layer's refusal direction
             # Sparsify it to be consistent with config
             raw_refusal_dir = measurement_results[f'refuse_{source_idx}']
             refusal_dir = sparsify_tensor(
                tensor=raw_refusal_dir,
                method=config.ablation.sparsify_method,
                threshold=config.ablation.magnitude_threshold if config.ablation.sparsify_method == "magnitude" else config.ablation.quantile
             ) 
             refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=0)

    # Determine Method Flags
    method = config.ablation.method
    use_biprojection_apply = method in ["biprojection", "full"]
    use_norm_preserving = method in ["norm_preserving", "full", "norm-preserving"]

    # Get Harmless Mean for Orthogonalization (Biprojection)
    if use_biprojection_apply:
        harmless_vec = measurement_results[f'harmless_{layer_idx}'].float()
        
        # Orthogonalize refusal_dir w.r.t local harmless_vec
        device = refusal_dir.device
        harmless_vec = harmless_vec.to(device)
        
        refusal_dir = remove_orthogonal_projection(refusal_dir, harmless_vec)
        # For stability, we might want to normalize again or not.
        # Original code normalized here:
        refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=0)

    return scale, refusal_dir, use_norm_preserving

def modify_shard_weights(
    state_dict: Dict[str, torch.Tensor],
    config: ModelConfig,
    global_refusal_dir: torch.Tensor,
    measurement_results: Dict[str, torch.Tensor]
) -> bool:
    """
    Modifies weights in the provided state_dict in-place.
    Returns True if any modification happened.
    """
    modified = False
    keys = list(state_dict.keys())
    
    for key in keys:
        if "layers." not in key:
            continue
        
        # Check if this is a target weight (o_proj or down_proj)
        parts = key.split(".")
        try:
            # Find the index after 'layers'
            layers_idx_in_parts = parts.index("layers")
            layer_idx = int(parts[layers_idx_in_parts + 1])
        except (ValueError, IndexError):
            continue

        target_modules = ["o_proj", "down_proj"] # Target modules
        is_target = any(m in key for m in target_modules) and key.endswith(".weight")
        
        if is_target:
            scale, refusal_dir, use_norm_preserving = get_layer_ablation_config(
                config, layer_idx, global_refusal_dir, measurement_results
            )

            # Apply Modification
            method_name = "norm_preserving" if use_norm_preserving else "simple"
            print(f"  Modifying {key} (Layer {layer_idx}) Scale={scale} Method={method_name}")
            
            W = state_dict[key]
            if use_norm_preserving:
                new_W = modify_tensor_norm_preserved(W, refusal_dir, scale_factor=scale)
            else:
                 new_W = modify_tensor_simple(W, refusal_dir, scale_factor=scale)
            state_dict[key] = new_W
            modified = True
            
    return modified

def run_sharded_ablation(
    config: ModelConfig,
    global_refusal_dir: torch.Tensor,
    measurement_results: dict
):
    """
    Executes the sharded ablation process.
    Iterates through model shards, applies ablation to target layers, and saves modified shards.
    """
    print("\n" + "="*60)
    print("PHASE 2: Sharded Ablation & Weight Modification")
    print("="*60)
    
    # Prepare Output
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve Paths
    index_path, model_dir, weight_map, shards = resolve_model_paths(config.model_id)
    
    # Process Shards
    for shard_file in tqdm(shards, desc="Processing Shards"):
        # Handle local vs cache paths
        if isinstance(model_dir, Path):
            shard_path = model_dir / shard_file
        else:
             shard_path = Path(model_dir) / shard_file
             
        print(f"Loading shard {shard_file}...")
        
        # Load Shard
        state_dict = load_file(str(shard_path))
        
        # Modify Shard
        is_modified = modify_shard_weights(state_dict, config, global_refusal_dir, measurement_results)

        if is_modified:
            print(f"  Saving modified shard {shard_file}...")
            save_file(state_dict, output_path / shard_file)
        else:
            print(f"  Copying unmodified shard {shard_file}...")
            # Use shutil copy for speed on unmodified shards
            shutil.copy(shard_path, output_path / shard_file)
            
        del state_dict
        gc.collect()

    # Finalize
    copy_model_artifacts(config, output_path, index_path)
