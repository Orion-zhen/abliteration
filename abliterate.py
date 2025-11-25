import gc
import os
import sys
import torch
import random
from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from utils.data import load_data
from utils.compute import compute_refusals, get_all_layer_means
from utils.apply import apply_abliteration
from utils.arguments import parser, generate_config


if __name__ == "__main__":
    args = parser.parse_args()
    config = generate_config(args)

    # Type assertions for critical config values
    assert isinstance(config["model"], str)
    assert isinstance(config["skip-begin"], int)
    assert isinstance(config["skip-end"], int)
    assert isinstance(config["scale-factor"], float)
    assert isinstance(config["refusal-quantile"], float)
    assert isinstance(config["refusal-top-k"], int)

    torch.inference_mode()
    torch.set_grad_enabled(False)

    if config["precision"] == "fp16":
        precision = torch.float16
    elif config["precision"] == "bf16":
        precision = torch.bfloat16
    else:
        precision = torch.float32

    # Configure Quantization
    if config["load-in-4bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
        )
    elif config["load-in-8bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    # Load Data
    if isinstance(config["data-harmful"], str):
        harmful_list = load_data(config["data-harmful"])
    else:
        harmful_list = load_data("./data/harmful.parquet")

    if isinstance(config["data-harmless"], str):
        harmless_list = load_data(config["data-harmless"])
    else:
        harmless_list = load_data("./data/harmless.parquet")

    if config["deccp"]:
        deccp_list = load_dataset("augmxnt/deccp", split="censored")
        harmful_list += deccp_list["text"]  # type: ignore

    if isinstance(config["num-harmful"], int) and config["num-harmful"] > 0:
        harmful_list = random.sample(harmful_list, config["num-harmful"])
    if isinstance(config["num-harmless"], int) and config["num-harmless"] > 0:
        harmless_list = random.sample(harmless_list, config["num-harmless"])

    print(f"Loaded model {config['model']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        trust_remote_code=True,
        dtype=precision,
        low_cpu_mem_usage=True,
        device_map=config["device"],
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if config["flash-attn"] else None,
    )
    model.requires_grad_(False)

    # Check layer bounds
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        total_layers = len(model.model.layers)
        if config["skip-begin"] + config["skip-end"] >= total_layers:
            raise ValueError("Too many layers to skip.")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], trust_remote_code=True, device_map=config["device"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # Refusal Direction & Harmless Means Logic
    # ---------------------------------------------------------
    global_refusal_dir = None
    all_harmless_means = None

    if isinstance(config["input-refusal"], str):
        print(f"Loading refusal data from {config['input-refusal']}...")
        loaded_data = torch.load(config["input-refusal"])

        if (
            isinstance(loaded_data, dict)
            and "refusal_dir" in loaded_data
            and "harmless_means" in loaded_data
        ):
            # Modern format
            global_refusal_dir = loaded_data["refusal_dir"]
            all_harmless_means = loaded_data["harmless_means"]
            print("Loaded refusal direction and harmless means.")
        elif isinstance(loaded_data, torch.Tensor):
            # Legacy format or just direction provided
            print("Loaded refusal direction (tensor only).")
            global_refusal_dir = loaded_data
            # We still need harmless means for Biprojection (orthogonalization).
            # If they are missing, we MUST compute them.
            print(
                "Harmless means not found in input file. Computing them now (required for Biprojection)..."
            )
            all_harmless_means = get_all_layer_means(
                model, tokenizer, harmless_list, batch_size=config.get("batch_size", 4)
            )
        else:
            raise ValueError("Invalid format for input-refusal file.")

    else:
        print("Computing refusal tensor and harmless means...")
        # This function now returns a tuple
        global_refusal_dir, all_harmless_means = compute_refusals(
            model=model,
            tokenizer=tokenizer,
            harmful_list=harmful_list,
            harmless_list=harmless_list,
            batch_size=config.get("batch_size", 4),
            refusal_quantile=config["refusal-quantile"],
            refusal_top_k=config["refusal-top-k"],
            plot_path=os.path.join(str(config["output"]), "refusal-scores.png")
        )

    # Save logic
    if isinstance(config["output-refusal"], str):
        print(f"Saving refusal data to {config['output-refusal']}...")
        # Save both as a dictionary for future reloading
        torch.save(
            {"refusal_dir": global_refusal_dir, "harmless_means": all_harmless_means},
            config["output-refusal"],
        )

    if not isinstance(config["output"], str):
        print("No output directory specified. Exiting after computation.")
        sys.exit(0)

    # ---------------------------------------------------------
    # Application Logic
    # ---------------------------------------------------------
    print("Applying abliteration...")

    # Explicit garbage collection before heavy lifting
    gc.collect()
    torch.cuda.empty_cache()

    # Reload model if it was loaded in 4bit/8bit (Standard practice in this repo to reload for modification)
    # Note: If we are just modifying weights in place and saving, we might need full precision model.
    if config["load-in-4bit"] or config["load-in-8bit"]:
        print("Reloading model with bf16 precision for weight modification...")
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # Load on CPU first to avoid OOM, apply_abliteration will handle moving parts to device if needed?
        # Ideally we load to 'cpu' or 'auto' if VRAM permits.
        # The original code loaded to "cpu" here.
        model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

    # Apply the Norm-Preserving Biprojected Abliteration
    model = apply_abliteration(
        model,
        global_refusal_dir,
        all_harmless_means,
        skip_begin_layers=config["skip-begin"],
        skip_end_layers=config["skip-end"],
        scale_factor=config["scale-factor"],
    )

    print(f"Saving abliterated model to {config['output']}...")
    model.save_pretrained(config["output"])
    tokenizer.save_pretrained(config["output"])
    print("Done.")
