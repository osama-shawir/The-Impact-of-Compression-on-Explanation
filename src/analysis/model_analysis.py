import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
import pickle
from src.utils.paths import OUTPUTS_DIR
from src.explainer.shapley_explainer import ShapleyExplainer

def generate_model_explanations(models, samples, fast_mode=False):
    """Generate SHAP explanations and predictions for NLP models"""
    cache_path = OUTPUTS_DIR / "model_analysis_results.pkl"
    
    # Check for cached results
    if cache_path.exists():
        print("Loading cached results...")
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
        return results["explanations"], results["predictions"], results["metrics"]
    
    print("Generating new explanations...")
    sample_texts = [str(text) for text in samples["Samples"]["text"]]
    true_labels = list(samples["Samples"]["label"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fast_mode:
        sample_texts = sample_texts[:2]
        true_labels = true_labels[:2]

    explanations_dict = {}
    predictions_dict = {}
    metrics_dict = {}

    for model_name, model_info in models.items():
        print(f"\nAnalyzing {model_name}...")
        model = model_info["model"].to(device)

        explainer = ShapleyExplainer(
            model,
            model_info["tokenizer"],
            max_length=128 if fast_mode else 512,
        )

        predictions = []
        for text in sample_texts:
            encoded = model_info["tokenizer"](
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128 if fast_mode else 512,
                return_attention_mask=True,
            )

            inputs = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits).cpu().item()
                predictions.append(pred)

        predictions_dict[model_name] = predictions
        metrics_dict[model_name] = classification_report(
            true_labels, predictions, output_dict=True, zero_division=1
        )

        shap_values = explainer.generate_explanations(sample_texts, fast_mode=fast_mode)
        explanations_dict[model_name] = shap_values

    # Cache results
    results = {
        "explanations": explanations_dict,
        "predictions": predictions_dict,
        "metrics": metrics_dict
    }
    
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)

    return explanations_dict, predictions_dict, metrics_dict