import gc
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def get_all_layer_means(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompts: list[str],
    batch_size: int = 4,
    max_length: int = 512,
) -> torch.Tensor:
    """Computes the mean activation value of the last token across all layers for a list of prompts.
    Uses batch processing and keeps accumulators on CPU.

    Args:
        model (PreTrainedModel): The LLM model
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The associated tokenizer
        prompts (list[str]): List of prompt strings
        batch_size (int, optional): Batch size for inference. Defaults to 4.
        max_length (int, optional): Max sequence length for tokenization. Defaults to 512.

    Returns:
        torch.Tensor: A tensor of shape [num_layers, hidden_size] containing the mean activation values.
    """

    device = model.device
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Force to right padding, crucial for extracting last token.
    if tokenizer.padding_side != "right":
        print(
            f"Warning: Tokenizer padding_side is '{tokenizer.padding_side}'. Forcing 'right' padding for correct activation extraction."
        )
        tokenizer.padding_side = "right"

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize accumulators on CPU
    # shape: [num_layers, hidden_size]
    activation_sums = torch.zeros(
        (num_layers, hidden_size), dtype=torch.float32, device="cpu"
    )
    total_count = 0

    # Create a simple DataLoader for batching
    data_loader = DataLoader(prompts, batch_size=batch_size, shuffle=False)  # type: ignore

    for batch_prompts in tqdm(data_loader, desc=f"Processing prompts"):
        # Apply chat template
        formatted_prompts = []
        for p in batch_prompts:
            if (
                hasattr(tokenizer, "chat_template")
                and tokenizer.chat_template is not None
            ):
                try:
                    # Construct message
                    messages = [{"role": "user", "content": p}]
                    # Apply template, DO NOT tokenize yet.
                    # add_generation_prompt=True is CRITICAL: it adds the assistant start token.
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted)
                except Exception as e:
                    # Fallback if template fails or model is base model
                    formatted_prompts.append(p)
            else:
                formatted_prompts.append(p)
        # Tokenize batch
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device=device)

        indices_expanded = None
        last_token_activations = None
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # Hidden states is a tuple of (batch_size, seq_len, hidden_size)
            # Index 0 is embedding layer
            # Transforers usually returns num_layers + 1 states. We focus on the actual layers (1 to N)
            # If the model output includes embedding as 0, we take [1:]
            all_hidden = outputs.hidden_states
            start_idx = 1 if len(all_hidden) > num_layers else 0

            # Robust Last Token Extraction
            # Find the last non-padding token index for each sequence in the batch
            # inputs.attention_mask: [batch, seq_len] with 1 for token, 0 for pad
            # We want the index of the last '1'.
            # Method:
            # 1. Flip the mask: [1, 1, 0] -> [0, 1, 1] (if calculating cumsum on flipped, complexity...)
            # 2. Simple way for Right Padding: sum(1) - 1.
            # 3. Robust way for Any Padding:
            #    sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
            #    But we forced padding_side="right" above, so sum(1)-1 is now safe.

            last_token_indices = inputs.attention_mask.sum(dim=1) - 1

            # Additional Safety Check: Ensure indices are not negative (empty prompts)
            last_token_indices = last_token_indices.clamp(min=0)

            current_batch_count = inputs.input_ids.shape[0]

            for layer_idx, layer_hidden in enumerate(all_hidden[start_idx:]):
                # layer_hidden: [batch, seq_len, hidden_size]

                # Correctly gather specific indices from the seq_len dimension
                # gather requires same number of dims.
                # expand indices: [batch, 1, hidden_size]
                indices_expanded = last_token_indices.view(-1, 1, 1).expand(
                    -1, 1, hidden_size
                )

                # Gather: [batch, 1, hidden_size] -> Squeeze -> [batch, hidden_size]
                last_token_activations = torch.gather(
                    layer_hidden, 1, indices_expanded
                ).squeeze(1)

                activation_sums[layer_idx] += (
                    last_token_activations.detach().cpu().float().sum(dim=0)
                )

            total_count += current_batch_count

            del (
                outputs,
                all_hidden,
                inputs,
                last_token_indices,
                last_token_activations,
                indices_expanded,
            )
            torch.cuda.empty_cache()

    # Compute mean
    if total_count == 0:
        raise ValueError("No prompts processed!")
    means = activation_sums / total_count
    return means


def compute_layer_scores(
    harmful_means: torch.Tensor, harmless_means: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Computes quanlity scores for each layer based on Signal-to-Noise Ratio (SNR) and Cosine Dissimilarity.

    Score = SNR * Dissimilarity
    SNR = ||r|| / max(||harmful_mean||, ||harmless_mean||)
    Dissimilarity = 1 - CosineSimilarity(harmful_mean, harmless_mean)

    Args:
        harmful_means (torch.Tensor): Mean activation values acorss harmful prompts. [num_layers, hidden_size]
        harmless_means (torch.Tensor): Mean activation values across harmless prompts. [num_layers, hidden_size]
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        torch.Tensor: Scores for each layer. [num_layers]
    """

    harmful_means = harmful_means.float()
    harmless_means = harmless_means.float()

    # Resusal directions
    refusal_dirs = harmful_means - harmless_means

    # Calculate Norms
    refusal_norms = torch.norm(refusal_dirs, dim=1)  # 为什么是 dim=1 ?
    harmful_norms = torch.norm(harmful_means, dim=1)
    harmless_norms = torch.norm(harmless_means, dim=1)

    # 1. Calculate SNR
    max_background_norms = torch.maximum(harmful_norms, harmless_norms)
    snr = refusal_norms / (max_background_norms + epsilon)

    # 2. Calculate Cosine Dissimilarity
    # Cosine Similarity = (A.B) / (||A|| * ||B||)
    dot_products = torch.sum(harmful_means * harmless_means, dim=1)
    cosine_sim = dot_products / (harmful_norms * harmless_norms + epsilon)
    dissimilarity = 1.0 - cosine_sim

    # 3. Compute Score
    scores = snr * dissimilarity
    print("Scores:")
    print(scores)

    return scores


def plot_scores(scores: torch.Tensor, top_indices: torch.Tensor, save_path: str) -> None:
    """
    Plots the refusal scores per layer and highlights the selected top layers.
    """

    # Move to CPU and numpy
    scores_np = scores.cpu().numpy()
    top_indices_np = top_indices.cpu().numpy()

    plt.figure(figsize=(12, 6))

    # Plot all scores
    plt.plot(
        range(len(scores_np)),
        scores_np,
        marker="o",
        markersize=3,
        label="Layer Score",
        color="royalblue",
    )

    # Highlight top K
    plt.scatter(
        top_indices_np,
        scores_np[top_indices_np],
        color="red",
        s=100,
        zorder=5,
        label="Selected Layers",
        edgecolors="white",
    )

    for idx in top_indices_np:
        plt.annotate(
            f"{idx}",
            (idx, scores_np[idx]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="red",
        )

    plt.title("Refusal Direction Quality Scores per Layer (SNR * Dissimilarity)")
    plt.xlabel("Layer Index")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Refusal score plot saved to {save_path}")


def apply_quantile_sparsification(
    vector: torch.Tensor, quantile: float = 0.995
) -> torch.Tensor:
    """Zeros out elements in the vector chat have absolute values below the specified quantile.

    Args:
        vector (torch.Tensor): Input tensor (1D of 2D).
        quantile (float, optional): The quantile threshold. Defaults to 0.995.

    Returns:
        torch.Tensor: Sparsified vector.
    """
    if quantile <= 0.0 or quantile >= 1.0:
        return vector

    # Calculate the threshold value
    abs_vector = torch.abs(vector)
    threshold = torch.quantile(abs_vector.float(), quantile)

    mask = abs_vector > threshold

    return vector * mask


def compute_refusals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    harmful_list: list[str],
    harmless_list: list[str],
    batch_size: int = 4,
    refusal_quantile: float = 0.995,
    refusal_top_k: int = 3,
    plot_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the global refusal direction and collects harmless means for all layers.

    Logic:

    1. Compute means for all layers for both harmful and harmless prompts.
    2. Score layers to find the best candidates for refusal extraction.
    3. Select Top-K layers.
    4. Compute refusal vectors for these layers.
    5. Apply magnitude sparsification.
    6. Average to get a robust global refusal direction.

    Args:
        model (PreTrainedModel): LLM model
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): The associated tokenizer.
        harmful_list (list[str]): List of harmful prompts.
        harmless_list (list[str]): List of harmless prompts.
        batch_size (int, optional): Batch size for inference. Defaults to 4.
        refusal_quantile (float, optional): Quantile for sparsification. Defaults to 0.995.
        refusal_top_k (int, optional): Number of top layers to aggregate. Defaults to 3.
        plot_path (str, optional): Path to save score figure. Defualts to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - global_refusal_dir [hidden_size]
            - all_harmless_means [num_layers, hidden_size]
    """

    print("Computing harmful means...")
    all_harmful_means = get_all_layer_means(
        model=model, tokenizer=tokenizer, prompts=harmful_list, batch_size=batch_size
    )

    print("Computing harmless means...")
    all_harmless_means = get_all_layer_means(
        model=model, tokenizer=tokenizer, prompts=harmless_list, batch_size=batch_size
    )

    # Calculate scores to find best layers
    scores = compute_layer_scores(
        harmful_means=all_harmful_means, harmless_means=all_harmless_means
    )

    # Select Top-K layers
    k = min(refusal_top_k, len(scores))
    top_scores, top_indices = torch.topk(scores, k=k)
    print(f"Top {k} layers selected for refusal extraction: {top_indices.tolist()}")
    print(f"Scores: {top_scores.tolist()}")
    
    if plot_path is not None:
        plot_scores(scores, top_indices, plot_path)

    # Compute and aggregate refusal directions
    # Extract raw refusal vectors from the top layers: r = harmful - harmless
    selected_harmful = all_harmful_means[top_indices]
    selected_harmless = all_harmless_means[top_indices]
    raw_refusal_dirs = selected_harmful - selected_harmless

    # Apply sparsification per vector
    sparse_refusal_dirs = torch.stack(
        [
            apply_quantile_sparsification(vec, refusal_quantile)
            for vec in raw_refusal_dirs
        ]
    )

    # Average them to get the global direction
    # shape: [hidden_size]
    global_refusal_dir = sparse_refusal_dirs.mean(dim=0)  # 为什么是 dim=0 ?

    # Normalize the direction
    global_refusal_dir = global_refusal_dir / (global_refusal_dir.norm() + 1e-8)

    del all_harmful_means
    gc.collect()

    return global_refusal_dir, all_harmless_means
