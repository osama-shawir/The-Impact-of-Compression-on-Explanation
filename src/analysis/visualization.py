import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from pathlib import Path
from src.utils.paths import VISUALIZATIONS_DIR


def visualize_explanations(
    explanations_dict, predictions_dict, models, samples, metrics_dict
):
    """Create visualizations and statistical analysis of model explanations"""
    for model_name in models.keys():
        model_dir = VISUALIZATIONS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Word Importance Visualization
        html_output = shap.plots.text(explanations_dict[model_name], display=False)
        with open(model_dir / "word_importance.html", "w") as file:
            file.write(html_output)

        # Feature Importance Plot
        plt.figure(figsize=(12, 8))
        shap_plot = shap.plots.bar(explanations_dict[model_name], show=False)
        plt.savefig(model_dir / "feature_importance.png", bbox_inches="tight", dpi=300)
        plt.close()

    # Compare models if exactly two
    if len(models) == 2:
        model_names = list(models.keys())
        generate_shap_comparison_plots(
            explanations_dict, model_names, VISUALIZATIONS_DIR
        )

    # colors = ["#FF0000", "#0000FF"]  # red and blue


def generate_shap_comparison_plots(explanations_dict, model_names, output_dir):
    """Generate comparison plots for SHAP values between models"""
    comparison_dir = output_dir / "model_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Get SHAP values for both models
    model1_shap = explanations_dict[model_names[0]]
    model2_shap = explanations_dict[model_names[1]]

    # 1. SHAP Value Distribution Comparison
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ["#FF0000", "#0000FF"]  # red and blue

    # Plot histograms with enhanced styling
    for i, (model_name, shap_vals) in enumerate(
        [(model_names[0], model1_shap), (model_names[1], model2_shap)]
    ):
        plt.hist(
            np.concatenate(shap_vals.values),
            bins=50,
            alpha=0.2,  # Increased transparency
            label=model_name,
            color=colors[i],
            edgecolor="black",
            linewidth=1,
        )

    # Enhance labels and title
    plt.xlabel("SHAP Value", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.title(
        "Distribution of SHAP Values Across Models",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Enhance legend
    plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, loc="upper right")

    # Add grid for better readability
    plt.grid(True, alpha=0.1, linestyle="--")

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save with white background
    plt.savefig(
        comparison_dir / "shap_distribution.png",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )
    plt.close()

    # 2. Feature Importance Correlation with padding
    def pad_and_mean(shap_values):
        # Get max length across all samples
        max_len = max(len(values) for values in shap_values.values)
        # Pad each sample to max length and compute mean
        padded_values = []
        for values in shap_values.values:
            padding_length = max_len - len(values)
            padded = np.pad(
                values, ((0, padding_length)), mode="constant", constant_values=0
            )
            padded_values.append(padded)
        return np.abs(np.array(padded_values)).mean(0)

    mean_shap1 = pad_and_mean(model1_shap)
    mean_shap2 = pad_and_mean(model2_shap)

    # Ensure equal lengths for correlation plot
    min_len = min(len(mean_shap1), len(mean_shap2))
    mean_shap1 = mean_shap1[:min_len]
    mean_shap2 = mean_shap2[:min_len]

    plt.figure(figsize=(10, 10))
    plt.scatter(mean_shap1, mean_shap2, alpha=0.5)
    plt.plot(
        [0, max(mean_shap1.max(), mean_shap2.max())],
        [0, max(mean_shap1.max(), mean_shap2.max())],
        "r--",
    )
    plt.xlabel(f"{model_names[0]} Mean |SHAP|")
    plt.ylabel(f"{model_names[1]} Mean |SHAP|")
    plt.title("Feature Importance Correlation")
    plt.savefig(
        comparison_dir / "feature_correlation.png", bbox_inches="tight", dpi=300
    )
    plt.close()

    # 3. Top Features Comparison
    feature_diff = mean_shap1 - mean_shap2

    # Get actual words as feature names from SHAP explanation object
    feature_names = []
    seen_features = set()

    # Collect unique feature names from all samples
    for sample in model1_shap.data:
        for word in sample:
            if word not in seen_features and isinstance(word, str):
                seen_features.add(word)
                feature_names.append(word)

    # Ensure feature_names matches the length of feature_diff
    feature_names = feature_names[: len(feature_diff)]  # Truncate if too long
    if len(feature_names) < len(feature_diff):  # Pad if too short
        feature_names.extend([""] * (len(feature_diff) - len(feature_names)))

    # Create DataFrame with actual words and differences
    diff_df = pd.DataFrame({"feature": feature_names, "difference": feature_diff})

    # Remove duplicates and sort by absolute difference
    diff_df["abs_diff"] = diff_df["difference"].abs()
    diff_df = diff_df.sort_values("abs_diff", ascending=False)
    diff_df = diff_df.drop_duplicates(subset=["feature"], keep="first")

    # Get top features
    top_features = min(20, len(diff_df))
    diff_df = diff_df.head(top_features)

    # Create plot with reversed order (largest at top)
    plt.figure(figsize=(12, 8))
    plt.barh(y=range(len(diff_df)), width=diff_df["difference"][::-1])
    plt.yticks(range(len(diff_df)), diff_df["feature"][::-1])
    plt.xlabel("SHAP Value Difference")
    plt.title(
        f"Top {top_features} Feature Importance Differences\n({model_names[0]} - {model_names[1]})"
    )
    plt.savefig(comparison_dir / "top_features_diff.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 4. Global Feature Rankings
    rank_comparison = pd.DataFrame(
        {
            model_names[0]: np.argsort(mean_shap1)[::-1],
            model_names[1]: np.argsort(mean_shap2)[::-1],
        }
    )
    rank_correlation = stats.spearmanr(
        rank_comparison[model_names[0]], rank_comparison[model_names[1]]
    )

    with open(comparison_dir / "rank_correlation.txt", "w") as f:
        f.write(f"Spearman Rank Correlation: {rank_correlation.correlation:.3f}\n")
        f.write(f"P-value: {rank_correlation.pvalue:.3f}")
