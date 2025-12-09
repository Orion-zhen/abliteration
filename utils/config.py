import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Any


@dataclass
class InferenceConfig:
    device: str = "cuda"
    batch_size: int = 4
    max_length: int = 512
    flash_attn: bool = False


@dataclass
class MeasurementsConfig:
    load_path: Optional[str] = None
    save_path: Optional[str] = None
    harmful_prompts: str = "./data/harmful.parquet"
    harmless_prompts: str = "./data/harmless.parquet"
    clip: float = 1.0


@dataclass
class AblationConfig:
    method: str = "full"
    sparsify_method: str = "percentile"
    quantile: float = 0.995
    magnitude_threshold: float = 0.05
    top_k: int = 10
    global_scale: float = 1.0
    layer_overrides: Dict[Union[int, str], Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    model: str
    output_dir: Optional[str]
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    measurements: MeasurementsConfig = field(default_factory=MeasurementsConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)


def load_config(config_path: str) -> ModelConfig:
    """Loads and validates the YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Basic Validation
    if "model" not in raw_config:
        raise ValueError("Config must contain 'model'")

    # Check for at least one output destination
    output_dir = raw_config.get("output_dir")
    measurements_save_path = raw_config.get("measurements", {}).get("save_path")

    if not output_dir and not measurements_save_path:
        raise ValueError(
            "Config must provide at least one of 'output_dir' or 'measurements.save_path'"
        )

    # Parse Inference Section
    inf_raw = raw_config.get("inference", {})
    # Check for typo in key from user request 'max_lengh'
    max_len = inf_raw.get("max_length", inf_raw.get("max_lengh", 512))

    inference_config = InferenceConfig(
        device=inf_raw.get("device", "cuda"),
        batch_size=inf_raw.get("batch_size", 4),
        max_length=max_len,
        flash_attn=inf_raw.get("flash_attn", False),
    )

    # Parse Measurements Section
    meas_raw = raw_config.get("measurements", {})
    measurements_config = MeasurementsConfig(
        load_path=meas_raw.get("load_path"),
        save_path=meas_raw.get("save_path"),
        harmful_prompts=meas_raw.get("harmful_prompts", "./data/harmful.parquet"),
        harmless_prompts=meas_raw.get("harmless_prompts", "./data/harmless.parquet"),
        clip=meas_raw.get("clip", 1.0),
    )

    # Parse Ablation Section
    ab_raw = raw_config.get("ablation", {})
    ablation_config = AblationConfig(
        method=ab_raw.get("method", "full"),
        sparsify_method=ab_raw.get("sparsify_method", "percentile"),
        quantile=ab_raw.get("quantile", 0.995),
        magnitude_threshold=ab_raw.get("magnitude_threshold", 0.05),
        top_k=ab_raw.get("top_k", 10),
        global_scale=ab_raw.get("global_scale", 1.0),
        layer_overrides=ab_raw.get("layer_overrides", {}),
    )

    # Main Config
    config = ModelConfig(
        model=raw_config["model"],
        output_dir=output_dir,  # Can be None
        inference=inference_config,
        measurements=measurements_config,
        ablation=ablation_config,
    )

    return config


from utils.output import Output


def print_config(config: ModelConfig):
    """Prints the configuration in a readable format."""
    Output.header("Configuration")

    Output.subheader("Model Settings")
    Output.key_value("Model Path", config.model)
    Output.key_value("Output Dir", config.output_dir or "N/A")

    Output.subheader("Inference")
    Output.key_value("Device", config.inference.device)
    Output.key_value("Batch Size", str(config.inference.batch_size))
    Output.key_value("Max Length", str(config.inference.max_length))
    Output.key_value("Flash Attention", str(config.inference.flash_attn))

    Output.subheader("Measurements")
    Output.key_value("Load Path", config.measurements.load_path or "N/A")
    Output.key_value("Save Path", config.measurements.save_path or "N/A")
    Output.key_value("Harmful Data", config.measurements.harmful_prompts)
    Output.key_value("Harmless Data", config.measurements.harmless_prompts)

    Output.subheader("Ablation Strategy")
    Output.key_value("Method", config.ablation.method)
    Output.key_value("Sparsify", config.ablation.sparsify_method)
    Output.key_value("Quantile", str(config.ablation.quantile))
    Output.key_value("Global Scale", str(config.ablation.global_scale))
    Output.key_value(
        "Layer Overrides", f"{len(config.ablation.layer_overrides)} layers"
    )
