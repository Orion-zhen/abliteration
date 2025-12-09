import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.output import Output


def analyze_results(
    results: dict, output_dir: str = ".", output_plot_name: str = "refusal_analysis.png"
):
    """
    Analyzes measurements and produces a chart similar to ref/analyze.py.
    """

    # Infer number of layers
    layers = 0
    while f"harmful_{layers}" in results:
        layers += 1

    if layers == 0 and "layers" in results:
        layers = results["layers"]

    Output.subheader(f"Analyzing {layers} Layers")

    cosine_similarities = []
    cosine_similarities_harmful = []
    cosine_similarities_harmless = []
    harmful_norms = []
    harmless_norms = []
    refusal_directions = []
    snratios = []
    signal_quality_estimates = []
    purity_ratios = []

    layer_data = []

    for layer in range(layers):
        harmful_mean = results[f"harmful_{layer}"].float()
        harmless_mean = results[f"harmless_{layer}"].float()
        refusal_dir = results[f"refuse_{layer}"].float()

        # 1. Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            harmful_mean, harmless_mean, dim=0
        ).item()
        cosine_similarities.append(cos_sim)

        cos_sim_harmful = torch.nn.functional.cosine_similarity(
            harmful_mean, refusal_dir, dim=0
        ).item()
        cosine_similarities_harmful.append(cos_sim_harmful)

        cos_sim_harmless = torch.nn.functional.cosine_similarity(
            harmless_mean, refusal_dir, dim=0
        ).item()
        cosine_similarities_harmless.append(cos_sim_harmless)

        # 2. Magnitudes
        harmful_norm = harmful_mean.norm().item()
        harmful_norms.append(harmful_norm)
        harmless_norm = harmless_mean.norm().item()
        harmless_norms.append(harmless_norm)

        # 3. Refusal direction properties
        refusal_norm = refusal_dir.norm().item()
        refusal_directions.append(refusal_norm)

        # 4. Signal-to-noise ratio
        snr = refusal_norm / max(harmful_norm, harmless_norm)
        snratios.append(snr)

        # 5. Refusal purity
        # Normalize the harmless direction for projection
        harmless_normalized = harmless_mean / harmless_mean.norm()

        # Project refusal onto harmless
        projection = (refusal_dir @ harmless_normalized) * harmless_normalized

        # Orthogonalized refusal direction (Gram-Schmidt)
        refusal_orth = refusal_dir - projection

        # Compute purity ratio
        purity_ratio = (refusal_orth.norm() / refusal_dir.norm()).item()
        purity_ratios.append(purity_ratio)

        # 6. Signal quality
        quality = snr * (1 - cos_sim) * purity_ratio
        signal_quality_estimates.append(quality)

        # Collect data for table
        layer_data.append(
            {
                "Layer": layer,
                "Quality": round(quality, 4),
                "SNR": round(snr, 4),
                "Purity": round(purity_ratio, 4),
                "CosSim": round(cos_sim, 4),
                "RefuseNorm": round(refusal_norm, 4),
            }
        )

    # Print Summary Table (Top 10 by Quality)
    Output.info("Top Layers by Estimated Signal Quality:")
    top_layers = sorted(layer_data, key=lambda x: x["Quality"], reverse=True)[:10]
    Output.table(
        top_layers,
        headers=["Layer", "Quality", "SNR", "Purity", "CosSim", "RefuseNorm"],
    )

    # Charting
    Output.info("Generating analysis charts...")
    layer_indices = range(layers)
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(
        "Refusal Direction Analysis Across Layers", fontsize=16, fontweight="bold"
    )

    # Plot 1: Mean Norms
    ax1 = axes[0, 0]
    ax1.plot(
        layer_indices,
        harmful_norms,
        "r-o",
        label="Harmful Mean",
        linewidth=2,
        markersize=4,
    )
    ax1.plot(
        layer_indices,
        harmless_norms,
        "g-s",
        label="Harmless Mean",
        linewidth=2,
        markersize=4,
    )
    ax1.plot(
        layer_indices,
        refusal_directions,
        "b-^",
        label="Refusal Direction",
        linewidth=2,
        markersize=4,
    )
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Norm", fontsize=11)
    ax1.set_title("Mean Norms vs Layer", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cosine Similarity
    ax2 = axes[0, 1]
    ax2.plot(
        layer_indices,
        cosine_similarities,
        "purple",
        label="Harmful to harmless",
        marker="o",
        linewidth=2,
        markersize=4,
    )
    ax2.plot(
        layer_indices,
        cosine_similarities_harmful,
        "red",
        label="Harmful to refusal",
        marker="o",
        linewidth=2,
        markersize=4,
    )
    ax2.plot(
        layer_indices,
        cosine_similarities_harmless,
        "blue",
        label="Harmless to refusal",
        marker="o",
        linewidth=2,
        markersize=4,
    )
    ax2.set_xlabel("Layer", fontsize=11)
    ax2.set_ylabel("Cosine Similarity", fontsize=11)
    ax2.set_title("Cosine Similarity vs Layer", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Signal-to-Noise Ratio, Purity Ratio
    ax3 = axes[1, 0]
    ax3.plot(
        layer_indices,
        snratios,
        "darkorange",
        label="Signal to noise",
        marker="d",
        linewidth=2,
        markersize=4,
    )
    ax3.plot(
        layer_indices,
        purity_ratios,
        "darkgreen",
        label="Refusal purity",
        marker="d",
        linewidth=2,
        markersize=4,
    )
    ax3.set_xlabel("Layer", fontsize=11)
    ax3.set_ylabel("Ratio", fontsize=11)
    ax3.set_title(
        "Signal-to-Noise and Refusal Purity Ratios vs Layer",
        fontsize=12,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Est. Signal Quality
    ax4 = axes[1, 1]
    ax4.plot(
        layer_indices,
        signal_quality_estimates,
        "teal",
        marker="*",
        linewidth=2,
        markersize=6,
    )
    ax4.set_xlabel("Layer", fontsize=11)
    ax4.set_ylabel("Est. Signal Quality", fontsize=11)
    ax4.set_title("Estimated Signal Quality vs Layer", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_plot_name)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    Output.success(f"Analysis chart saved to {output_path}")
