import json
import shutil
import torch
import os
import pandas
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers.utils import cached_file

from utils.config import ModelConfig

def resolve_model_paths(model_id: str) -> Tuple[Optional[str], Union[Path, str], Dict[str, str], List[str]]:
    """
    Resolves model paths, handling both sharded and single-file models.
    Returns: index_path, model_dir, weight_map, shards
    """
    index_path = cached_file(model_id, "model.safetensors.index.json")
    model_dir = None
    weight_map = {}
    
    if not index_path:
        # Fallback for non-sharded models (single file)
        single_file = cached_file(model_id, "model.safetensors")
        if single_file:
            print("Model is not sharded (single file). Treating as single shard.")
            weight_map = {"model.safetensors": "model.safetensors"}
            model_dir = Path(single_file).parent
        else:
             # Try checking if it's a local directory that doesn't use cache
            local_path = Path(model_id)
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

    shards = sorted(list(set(weight_map.values())))
    return index_path, model_dir, weight_map, shards

def copy_model_artifacts(config: ModelConfig, output_path: Path, index_path: Optional[str]):
    """
    Copies configuration and index files to the output directory.
    """
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
            path = cached_file(config.model, filename)
            if path and Path(path).exists():
                shutil.copy(path, output_path / filename)
            else:
                # Check local if cached_file returns None or fail
                # But cached_file handles local paths too.
                # If model is local path, check directly
                if (Path(config.model) / filename).exists():
                     shutil.copy(Path(config.model) / filename, output_path / filename)
        except Exception as e:
            # Just log and skip
            print(f"  Skipping {filename}")
            pass

def save_measurements(results: dict, layer_scores: dict, path: str):
    """Saves raw measurements and scores to a file."""
    print(f"Saving measurements to {path}...")
    # Using torch.save for simplicity as it handles dictionary of tensors well
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"results": results, "layer_scores": layer_scores}
    torch.save(data, path)
    print("Measurements saved.")

def load_measurements(path: str) -> tuple[dict, dict]:
    """Loads measurements from a file."""
    print(f"Loading measurements from {path}...")
    if not os.path.exists(path):
         raise FileNotFoundError(f"Measurements file not found: {path}")
    
    data = torch.load(path, weights_only=False) # weights_only=False primarily for dict structure support, trusted source assumption
    # In newer torch, weights_only=True is default. If it's just tensors in dict, it might work, but safer to warn/handle.
    # Given we are saving a dict of tensors and a dict of floats, standard torch.save/load is fine.
    
    results = data["results"]
    layer_scores = data["layer_scores"]
    print("Measurements loaded.")
    return results, layer_scores

def load_data(path: str) -> list[str]:
    if path.endswith(".txt"):
        with open(path, "r") as f:
            return f.readlines()
    elif path.endswith(".parquet"):
        df = pandas.read_parquet(path)
        data = df.get("text")
        if data is None:
            raise ValueError("No 'text' column found in parquet file")
        return data.tolist()
    elif path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list")
        return data
    else:
        raise ValueError("Unsupported file format")
