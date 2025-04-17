import torch
import numpy as np
from src.models.model_loader import load_models
from src.data.data_loader import get_imdb_samples
from src.analysis.model_analysis import generate_model_explanations
from src.analysis.visualization import visualize_explanations
from src.utils.paths import OUTPUTS_DIR

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


def main(fast_mode=False):
    # Load models and data
    models = load_models()

    # Get small sample for SHAP analysis
    samples_small = get_imdb_samples(n_samples=10, show_examples=3)

    # Generate or load explanations
    explanations_dict, predictions_dict, metrics_dict = generate_model_explanations(
        models, samples_small, fast_mode=fast_mode
    )

    # Visualize SHAP results
    visualize_explanations(
        explanations_dict, predictions_dict, models, samples_small, metrics_dict
    )

    # Get larger sample for performance comparison
    if not fast_mode:
        print("\nGenerating performance comparison with larger dataset...")
        samples_large = get_imdb_samples(n_samples=1001, show_examples=0)
        from src.analysis.model_performance import generate_performance_comparison

        performance_df = generate_performance_comparison(models, samples_large)
        print(
            "\nPerformance comparison complete. Results saved to outputs/visualizations/model_comparison/"
        )


if __name__ == "__main__":
    main(fast_mode=False)  # Set to False to run performance comparison
