from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Output directories
OUTPUTS_DIR = ROOT_DIR / "outputs"
EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"
METRICS_DIR = OUTPUTS_DIR / "metrics"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
MODEL_COMPARISON_DIR = VISUALIZATIONS_DIR / "model_comparison"

# Create all directories
directories = [
    OUTPUTS_DIR,
    EXPLANATIONS_DIR,
    METRICS_DIR,
    VISUALIZATIONS_DIR,
    MODEL_COMPARISON_DIR
]

for dir_path in directories:
    dir_path.mkdir(parents=True, exist_ok=True)