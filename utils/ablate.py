import gc
import json
import shutil
import torch
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from transformers.utils import cached_file

from utils.config import ModelConfig
from utils.modifier import modify_tensor_norm_preserved, modify_tensor_simple
from utils.sparsify import percentile_sparsify, sparsify_vector
from utils.vector_utils import remove_orthogonal_projection

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

    # Get Index
    index_path = cached_file(config.model_id, "model.safetensors.index.json")
    if not index_path:
        # Fallback for non-sharded models (single file)
        single_file = cached_file(config.model_id, "model.safetensors")
        if single_file:
            print("Model is not sharded (single file). Treating as single shard.")
            weight_map = {"model.safetensors": "model.safetensors"}
            model_dir = Path(single_file).parent
            # Also copy the single file index equivalent logic if needed, but for single file we loop once.
        else:
             # Try checking if it's a local directory that doesn't use cache
            local_path = Path(config.model_id)
            if (local_path / "model.safetensors.index.json").exists():
                 index_path = str(local_path / "model.safetensors.index.json")
                 model_dir = local_path
            elif (local_path / "model.safetensors").exists():
                 print("Found local single safetensors file.")
                 weight_map = {"model.safetensors": "model.safetensors"}
                 model_dir = local_path
                 index_path = None
            else:
                 raise ValueError("Could not find model.safetensors.index.json or model.safetensors")
    
    if index_path:
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        model_dir = Path(index_path).parent

    # Detect Layer Prefix
    layer_prefix = None
    for key in weight_map.keys():
        if ".layers." in key and ".self_attn." in key:
            layer_prefix = key.split(".layers.")[0]
            print(f"Detected layer prefix: {layer_prefix}")
            break
    if not layer_prefix:
         # Fallback for some models where layers might be just "layers"
         # But usually it is "model.layers" or "transformer.layers"
         # If we can't find it, we might be in trouble.
         # Check for any key with ".layers."
         cand = [k for k in weight_map.keys() if ".layers." in k]
         if cand:
             layer_prefix = cand[0].split(".layers.")[0]
             print(f"Detected layer prefix (fallback): {layer_prefix}")
         else:
             raise ValueError("Could not detect layer structure.")

    # Identify distinct shards
    shards = sorted(list(set(weight_map.values())))
    
    for shard_file in tqdm(shards, desc="Processing Shards"):
        # Handle local vs cache paths
        if isinstance(model_dir, Path):
            shard_path = model_dir / shard_file
        else:
             # This case shouldn't happen if logic above is correct
             shard_path = Path(model_dir) / shard_file
             
        print(f"Loading shard {shard_file}...")
        
        # Load Shard
        state_dict = load_file(str(shard_path))
        modified = False
        
        # Iterate keys in this shard
        keys = list(state_dict.keys()) # Copy keys
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
                # Resolve Configuration for this layer
                scale = config.ablation.global_scale
                refusal_vec = global_refusal_dir
                
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
                         raw_vec = measurement_results[f'refuse_{source_idx}']
                         refusal_vec = sparsify_vector(
                            raw_vec,
                            method=config.refusal.sparsify_method,
                            threshold=config.refusal.magnitude_threshold if config.refusal.sparsify_method == "magnitude" else config.refusal.quantile
                         ) 
                         refusal_vec = torch.nn.functional.normalize(refusal_vec, dim=0)

                # Determine Method Flags
                method = config.ablation.method
                use_biprojection_apply = method in ["biprojection", "full"]
                use_norm_preserving = method in ["norm_preserving", "full", "norm-preserving"]

                # Get Harmless Mean for Orthogonalization (Biprojection)
                # Ensure harmless_vec is same device/dtype
                harmless_vec = measurement_results[f'harmless_{layer_idx}'].float()
                        
                if use_biprojection_apply:
                    # Orthogonalize refusal_vec w.r.t local harmless_vec
                     device = refusal_vec.device
                     harmless_vec = harmless_vec.to(device)
                     
                     harmless_vec = harmless_vec.to(device)
                     
                     refusal_vec = remove_orthogonal_projection(refusal_vec, harmless_vec)
                     # For stability, we might want to normalize again or not.
                     # Original code normalized here:
                     refusal_vec = torch.nn.functional.normalize(refusal_vec, dim=0)

                # Apply Modification
                print(f"  Modifying {key} (Layer {layer_idx}) Scale={scale} Method={method}")
                W = state_dict[key]
                if use_norm_preserving:
                    new_W = modify_tensor_norm_preserved(W, refusal_vec, scale_factor=scale)
                else:
                     new_W = modify_tensor_simple(W, refusal_vec, scale_factor=scale)
                state_dict[key] = new_W
                modified = True
                
        if modified:
            print(f"  Saving modified shard {shard_file}...")
            save_file(state_dict, output_path / shard_file)
        else:
            print(f"  Copying unmodified shard {shard_file}...")
            # Use shutil copy for speed on unmodified shards
            shutil.copy(shard_path, output_path / shard_file)
            
        del state_dict
        gc.collect()

    # Final Config Copying
    print("\n" + "="*60)
    print("PHASE 3: Finalizing")
    print("="*60)
    print("Copying configuration files...")
    
    if index_path:
        shutil.copy(index_path, output_path / "model.safetensors.index.json")
    
    config_files = [
        "config.json", "generation_config.json", 
        "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json", "preprocessor_config.json",
        "chat_template.json"
    ]
    
    for filename in config_files:
        try:
            # Wrap in try-except to simple skip missing files
            path = cached_file(config.model_id, filename)
            if path and Path(path).exists():
                shutil.copy(path, output_path / filename)
            else:
                # Check local if cached_file returns None or fail
                # But cached_file handles local paths too.
                # If model_id is local path, check directly
                if (Path(config.model_id) / filename).exists():
                     shutil.copy(Path(config.model_id) / filename, output_path / filename)
        except Exception as e:
            # Just log and skip
            print(f"  Skipping {filename}: {e}")
            pass
