
import numpy as np
import torch
import shap
import scipy as sp

class ShapleyExplainer:
    def __init__(self, model, tokenizer, max_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_predict_fn(self):
        def f(x):
            # Ensure input is list of strings
            if isinstance(x, str):
                x = [x]
            elif not isinstance(x, list):
                x = list(x)

            # Properly encode with attention mask
            encoded = self.tokenizer(
                x,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            inputs = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)[0].cpu().numpy()
            scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
            val = sp.special.logit(scores[:, 1])
            return val

        return f

    def generate_explanations(self, texts, fast_mode=False):
        """Generate SHAP explanations with dynamic max_evals"""
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]

        batch_size = 2 if fast_mode else 32

        sample_encoding = self.tokenizer(
            texts[0],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
        )
        num_features = len(sample_encoding["input_ids"])
        min_max_evals = 2 * num_features + 1

        explainer = shap.Explainer(
            self.prepare_predict_fn(),
            self.tokenizer,
            algorithm="auto" if fast_mode else "permutation",
        )

        shap_values = explainer(
            texts,
            batch_size=batch_size,
            max_evals=min_max_evals if fast_mode else max(min_max_evals, 2000),
        )
        return shap_values