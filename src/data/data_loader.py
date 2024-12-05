from datasets import load_dataset
import random


def get_imdb_samples(n_samples=10, show_examples=5):
    """
    Get IMDB samples with their labels, supporting odd number of samples
    Returns: Dictionary with balanced(±1) positive/negative samples
    """
    dataset = load_dataset("imdb")
    test_data = dataset["test"]

    # Get indices for each class
    pos_indices = [i for i, label in enumerate(test_data["label"]) if label == 1]
    neg_indices = [i for i, label in enumerate(test_data["label"]) if label == 0]

    # Handle odd numbers
    n_pos = (n_samples + 1) // 2  # Positive gets extra sample if odd
    n_neg = n_samples - n_pos  # Negative gets remaining

    random.seed(42)
    selected_pos = random.sample(pos_indices, n_pos)
    selected_neg = random.sample(neg_indices, n_neg)

    # Rest of the function remains the same
    all_indices = selected_pos + selected_neg
    random.shuffle(all_indices)

    limited_dataset = {"Samples": test_data.select(all_indices)}

    # Print examples
    print("\nExample reviews and labels:")
    for i in range(min(show_examples, n_samples)):
        text = limited_dataset["Samples"]["text"][i][:100] + "..."
        label = (
            "Positive" if limited_dataset["Samples"]["label"][i] == 1 else "Negative"
        )
        print(f"\nReview {i+1}:")
        print(f"Text: {text}")
        print(f"Label: {label}")

    return limited_dataset
