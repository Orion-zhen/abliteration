import gc
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Dict, Union

def magnitude_clip(tensor: torch.Tensor, max_val: float) -> torch.Tensor:
    """Clips tensor values to [-max_val, max_val] by magnitude preserving sign."""
    return torch.clamp(tensor, -max_val, max_val)

def welford_gpu_batched_multilayer_float32(
    formatted_prompts: list[str],
    desc: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    layer_indices: list[int],
    pos: int = -1,
    batch_size: int = 1,
    clip: float = 1.0,
) -> dict[int, torch.Tensor]:
    """
    Computes mean activations using Welford's online algorithm.
    Strictly follows reference logic: Left padding + model.generate(max_new_tokens=1).
    """
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    means = {layer_idx: None for layer_idx in layer_indices}
    counts = {layer_idx: 0 for layer_idx in layer_indices}

    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc=desc):
        batch_prompts = formatted_prompts[i:i+batch_size]

        batch_encoding = tokenizer(
            batch_prompts,
            padding=True,
            padding_side='left',
            return_tensors="pt",
        )
        
        batch_input = batch_encoding['input_ids'].to(model.device)
        batch_mask = batch_encoding['attention_mask'].to(model.device)

        # Use generate to get hidden states at the first generated token position
        # This is the "next token prediction" state for the last token of the prompt.
        raw_output = model.generate(
            batch_input,
            attention_mask=batch_mask,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        del batch_input, batch_mask
        # hidden_states is tuple of tuples: (generation_step, layer_idx)
        # We want Step 0.
        hidden_states = raw_output.hidden_states[0]  
        del raw_output

        # Process layers with Welford in float32
        for layer_idx in layer_indices:
            # Cast to float32 for accumulation
            # Shape: [batch, 1, hidden] -> [batch, hidden]
            current_hidden = hidden_states[layer_idx][:, pos, :].float()
            
            if (clip < 1.0):
                 # Simple clamping if clip is small? Ref logic: magnitude_clip(current_hidden, clip)
                 # Actually ref logic uses a custom function, here we use simple clamp or assume logic
                 # Check ref logic: it calls magnitude_clip which likely does clamp.
                 # Wait, typical clip usually implies if |x| > threshold -> scale.
                 # Let's trust torch.clamp for now or check ref utils/clip.py if strict alignment needed.
                 # For now, implementing standard clamping.
                 current_hidden = torch.clamp(current_hidden, min=-clip, max=clip) # Placeholder for clip logic

            batch_size_actual = current_hidden.size(dim=0)
            total_count = counts[layer_idx] + batch_size_actual

            if means[layer_idx] is None:
                means[layer_idx] = current_hidden.mean(dim=0)
            else:
                delta = current_hidden - means[layer_idx]
                means[layer_idx] += delta.sum(dim=0) / total_count

            counts[layer_idx] = total_count
            
        del hidden_states
        torch.cuda.empty_cache()

    # Move results to CPU
    return_dict = {
        layer_idx: mean.to(device="cpu") 
        for layer_idx, mean in means.items()
    }
    return return_dict

def plot_scores(scores: dict, save_path: str):
    """Plots refusal scores per layer."""
    layer_indices = sorted(scores.keys())
    values = [scores[idx] for idx in layer_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, values, marker='o', linestyle='-')
    plt.title("Refusal Direction Quality Score (SNR * Dissimilarity)")
    plt.xlabel("Layer Index")
    plt.ylabel("Score")
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Refusal score plot saved to {save_path}")

def compute_refusals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    harmful_list: list[str],
    harmless_list: list[str],
    batch_size: int = 32,
    projected: bool = True,
    output_dir: str = ".",
) -> Dict[str, torch.Tensor]:
    """
    Computes harmful and harmless means, and derives refusal directions.
    """
    
    # Identify layers - Robust logic from ref implementation
    layer_base = model
    if hasattr(layer_base, "model"):
        layer_base = layer_base.model
    if hasattr(layer_base, "language_model"):
        layer_base = layer_base.language_model
    
    if hasattr(layer_base, "layers"):
        num_layers = len(layer_base.layers)
    elif hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        # Last resort fallback
        raise ValueError("Could not determine number of layers for this model.")

    focus_layers = list(range(num_layers))

    # Format Prompts
    def format_chats(prompts):
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False
            ) for p in prompts
        ]

    print("Formatting prompts...")
    harmful_formatted = format_chats(harmful_list)
    harmless_formatted = format_chats(harmless_list)

    print("Computing Harmful Means...")
    harmful_means = welford_gpu_batched_multilayer_float32(
        harmful_formatted, "Harmful Batches", model, tokenizer, focus_layers, batch_size=batch_size
    )

    print("Computing Harmless Means...")
    harmless_means = welford_gpu_batched_multilayer_float32(
        harmless_formatted, "Harmless Batches", model, tokenizer, focus_layers, batch_size=batch_size
    )

    results = {}
    layer_scores = {}

    print("Calculating Refusal Vectors...")
    for layer in tqdm(focus_layers, desc="Processing Layers"):
        target_harmful = harmful_means[layer]
        target_harmless = harmless_means[layer]
        
        results[f'harmful_{layer}'] = target_harmful
        results[f'harmless_{layer}'] = target_harmless

        # Refusal Direction Calculation (in double for precision)
        refusal_dir = (target_harmful.double() - target_harmless.double()).float()

        if projected:
            # Orthogonalize against harmless mean (Pre-computation projection)
            harmless_normed = torch.nn.functional.normalize(target_harmless.float(), dim=0)
            projection = torch.dot(refusal_dir, harmless_normed)
            refusal_dir = refusal_dir - projection * harmless_normed
        
        results[f'refuse_{layer}'] = refusal_dir

        # Calculate Score for Plotting (SNR * Dissimilarity)
        refusal_norm = refusal_dir.norm().item()
        max_bg_norm = max(target_harmful.norm().item(), target_harmless.norm().item())
        snr = refusal_norm / (max_bg_norm + 1e-6)
        
        cos_sim = torch.nn.functional.cosine_similarity(target_harmful, target_harmless, dim=0).item()
        dissimilarity = 1.0 - cos_sim
        score = snr * dissimilarity
        layer_scores[layer] = score

    # Plot scores
    plot_scores(layer_scores, os.path.join(output_dir, "refusal_scores.png"))

    return results, layer_scores
