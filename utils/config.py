import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Any

@dataclass
class DataConfig:
    harmful_path: str = "./data/harmful.parquet"
    harmless_path: str = "./data/harmless.parquet"
    batch_size: int = 4
    max_length: int = 512
    # For compatibility with ref logic
    clip: float = 1.0

@dataclass
class RefusalConfig:
    quantile: float = 0.995
    top_k: int = 10
    # projected: bool = True  # Removed
    # Measurements I/O
    measurements_save_path: Optional[str] = None
    measurements_load_path: Optional[str] = None
    # Sparsification Strategy
    sparsify_method: str = "percentile"  # "percentile", "magnitude"
    magnitude_threshold: float = 0.05
    # Percentile strategy uses 'quantile' field above

@dataclass
class AblationConfig:
    # Options: "simple", "biprojection", "norm-preserving", "full"
    # simple: No biprojection, No norm-preserving
    # biprojection: Biprojection, No norm-preserving
    # norm-preserving: No biprojection, Norm-preserving
    # full: Biprojection, Norm-preserving
    method: str = "full" 
    global_scale: float = 1.0
    # Overrides: layer_idx -> {source_layer: int, scale: float}
    layer_overrides: Dict[Union[int, str], Dict[str, Any]] = field(default_factory=dict)

@dataclass
class ModelConfig:
    model_id: str
    output_dir: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    flash_attn: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    refusal: RefusalConfig = field(default_factory=RefusalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

def load_config(config_path: str) -> ModelConfig:
    """Loads and validates the YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Basic Validation and Parsing
    if "model_id" not in raw_config:
        raise ValueError("Config must contain 'model_id'")
    if "output_dir" not in raw_config:
        raise ValueError("Config must contain 'output_dir'")

    # Parse Data Section
    data_raw = raw_config.get("data", {})
    data_config = DataConfig(
        harmful_path=data_raw.get("harmful_path", "./data/harmful.parquet"),
        harmless_path=data_raw.get("harmless_path", "./data/harmless.parquet"),
        batch_size=data_raw.get("batch_size", 4),
        max_length=data_raw.get("max_length", 512),
        clip=data_raw.get("clip", 1.0)
    )

    # Parse Refusal Calculation Section
    ref_raw = raw_config.get("refusal_calculation", {})
    refusal_config = RefusalConfig(
        quantile=ref_raw.get("quantile", 0.995),
        top_k=ref_raw.get("top_k", 10),
        # projected=ref_raw.get("projected", True), -> Removed, handled by ablation method now
        measurements_save_path=ref_raw.get("measurements_save_path"),
        measurements_load_path=ref_raw.get("measurements_load_path"),
        sparsify_method=ref_raw.get("sparsify_method", "percentile"),
        magnitude_threshold=ref_raw.get("magnitude_threshold", 0.05)
    )

    # Parse Ablation Section
    ab_raw = raw_config.get("ablation", {})
    ablation_config = AblationConfig(
        method=ab_raw.get("method", "full"),
        global_scale=ab_raw.get("global_scale", 1.0),
        layer_overrides=ab_raw.get("layer_overrides", {})
    )

    # Main Config
    config = ModelConfig(
        model_id=raw_config["model_id"],
        output_dir=raw_config["output_dir"],
        device=raw_config.get("device", "cuda"),
        dtype=raw_config.get("dtype", "bfloat16"),
        flash_attn=raw_config.get("flash_attn", True),
        data=data_config,
        refusal=refusal_config,
        ablation=ablation_config
    )

    return config

def print_config(config: ModelConfig):
    """Prints the configuration in a readable format."""
    print("=" * 60)
    print(f"Model ID: {config.model_id}")
    print(f"Output Dir: {config.output_dir}")
    print(f"Device: {config.device} ({config.dtype})")
    print("-" * 60)
    print("Data Config:")
    print(f"  Harmful: {config.data.harmful_path}")
    print(f"  Harmless: {config.data.harmless_path}")
    print(f"  Batch Size: {config.data.batch_size}")
    print("-" * 60)
    print("Refusal Calculation:")
    print(f"  Quantile: {config.refusal.quantile}")
    print(f"  Top K Layers: {config.refusal.top_k}")
    # print(f"  Projected (Measurement): {config.refusal.projected}") -> Removed
    print(f"  Measurements Load Path: {config.refusal.measurements_load_path}")
    print(f"  Measurements Save Path: {config.refusal.measurements_save_path}")
    print("-" * 60)
    print("Ablation:")
    print(f"  Global Scale: {config.ablation.global_scale}")
    print(f"  Overrides: {len(config.ablation.layer_overrides)} layers")
    print("=" * 60)
