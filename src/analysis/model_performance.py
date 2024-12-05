import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from src.utils.paths import VISUALIZATIONS_DIR, MODEL_COMPARISON_DIR, OUTPUTS_DIR
from tqdm import tqdm
import pickle
from pathlib import Path


def get_predictions_in_batches(model, tokenizer, texts, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    predictions = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            return_attention_mask=True,
        )

        inputs = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            predictions.extend(preds)

    return predictions


def generate_performance_comparison(models, dataset):
    cache_path = OUTPUTS_DIR / "performance_predictions_cache.pkl"

    # Try to load cached results
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
            if (
                cached_data["texts"] == dataset["Samples"]["text"]
                and cached_data["labels"] == dataset["Samples"]["label"]
            ):
                print("Using cached predictions...")
                df = cached_data["df"]

                markdown_table = df.to_markdown(index=False)
                with open(MODEL_COMPARISON_DIR / "performance_comparison.md", "w") as f:
                    f.write(markdown_table)

                # Create visualization
                plt.figure(figsize=(12, 6))
                df_melted = df.melt(
                    id_vars=["Model", "Class"],
                    value_vars=["precision", "recall", "f1-score"],
                    var_name="Metric",
                    value_name="Score",
                )

                sns.barplot(
                    data=df_melted,
                    x="Metric",
                    y="Score",
                    hue="Model",
                    palette=["#FF0000", "#0000FF"],
                    alpha=0.6,
                )

                plt.title("Model Performance Comparison")
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(
                    MODEL_COMPARISON_DIR / "performance_comparison.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                return df

    texts = dataset["Samples"]["text"]
    true_labels = dataset["Samples"]["label"]

    results = {}
    for model_name, model_info in models.items():
        print(f"\nGenerating predictions for {model_name}...")
        predictions = get_predictions_in_batches(
            model_info["model"], model_info["tokenizer"], texts
        )
        results[model_name] = classification_report(
            true_labels,
            predictions,
            output_dict=True,
            target_names=["Negative", "Positive"],
        )

    # Create comparison DataFrame
    metrics = ["precision", "recall", "f1-score"]
    classes = ["Negative", "Positive"]

    comparison_data = []
    for model_name, report in results.items():
        for class_name in classes:
            row = {"Model": model_name, "Class": class_name}
            for metric in metrics:
                row[metric] = report[class_name][metric]
            comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Cache the results
    cache_data = {"texts": texts, "labels": true_labels, "df": df}
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    # Generate markdown table
    markdown_table = df.to_markdown(index=False)
    with open(MODEL_COMPARISON_DIR / "performance_comparison.md", "w") as f:
        f.write("# Model Performance Comparison\n\n")
        f.write(markdown_table)

    # Create visualization
    plt.figure(figsize=(12, 6))
    df_melted = df.melt(
        id_vars=["Model", "Class"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    sns.barplot(
        data=df_melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#FF0000", "#0000FF"],
        alpha=0.6,
    )

    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        MODEL_COMPARISON_DIR / "performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return df
