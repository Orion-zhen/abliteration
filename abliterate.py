import argparse
import gc
import torch
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.config import load_config, print_config
from utils.measure import compute_refusals, save_measurements, load_measurements, inlayer_results_projection
from utils.data import load_data
from utils.sparsify import percentile_sparsify, sparsify_tensor
from utils.ablate import run_sharded_ablation
from utils.analyze import analyze_results

def main():
    parser = argparse.ArgumentParser(description="End-to-End Sharded Abliteration")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    print_config(config)

    # 2. Measurement Phase
    print("\n" + "="*60)
    print("PHASE 1: Measurement & Refusal Calculation")
    print("="*60)

    # Check if we should load measurements
    if config.refusal.measurements_load_path:
        results, layer_scores = load_measurements(config.refusal.measurements_load_path)
    else:
        # Load Model for Inference
        print(f"Loading model {config.model_id} for measurement...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=getattr(torch, config.dtype),
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2" if config.flash_attn else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        
        # Load Data
        harmful_data = load_data(config.data.harmful_path)
        harmless_data = load_data(config.data.harmless_path)
        
        # Compute Refusals (Raw)
        results, layer_scores = compute_refusals(
            model=model,
            tokenizer=tokenizer,
            harmful_list=harmful_data,
            harmless_list=harmless_data,
            batch_size=config.data.batch_size,
            # projected=False, -> Removed argument
            output_dir=config.output_dir
        )
        
        # Save Measurements if requested
        if config.refusal.measurements_save_path:
            save_measurements(results, layer_scores, config.refusal.measurements_save_path)
        
        # Unload Model
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded. Memory cleared.")

    # Apply Projection if requested (Method based)
    if config.ablation.method in ["biprojection", "full"]:
        inlayer_results_projection(results)
    
    # Analyze Refusal Directions
    analyze_results(results, config.output_dir)
    
    # Calculate Global Refusal Direction (Top-K Average)
    sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_indices = [x[0] for x in sorted_layers[:config.refusal.top_k]]
    print(f"Refusal Calculation: Selected Top-{config.refusal.top_k} layers: {top_k_indices}")

    # Gather refusal vectors from top-k layers
    selected_refusals = []
    for idx in top_k_indices:
        local_refusal_dir = results[f'refuse_{idx}']
        # Use configured sparsify strategy
        sparsed_local_refusal_dir = sparsify_tensor(
            local_refusal_dir, 
            method=config.refusal.sparsify_method,
            threshold=config.refusal.magnitude_threshold if config.refusal.sparsify_method == "magnitude" else config.refusal.quantile,
            k=config.refusal.top_k # Logic reuse top_k for topk method if used, but user asked for mag/per
        )
        selected_refusals.append(sparsed_local_refusal_dir)
    
    global_refusal_dir = torch.stack(selected_refusals).mean(dim=0)
    global_refusal_dir = torch.nn.functional.normalize(global_refusal_dir, dim=0)
    
    print("Global refusal direction computed.")



    # 3. Sharded Ablation Phase
    run_sharded_ablation(
        config=config,
        global_refusal_dir=global_refusal_dir,
        measurement_results=results
    )
    
    # Save config details
    output_config_path = f"{config.output_dir}/abliteration_config.yaml"
    shutil.copy(args.config, output_config_path)
    
    print(f"Done! Abliterated model saved to {config.output_dir}")

if __name__ == "__main__":
    main()
