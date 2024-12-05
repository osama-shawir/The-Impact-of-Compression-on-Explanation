
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_models():
    """
    Load different pre-trained BERT model variants for sentiment analysis from HuggingFace
    """
    models = {
        "bert_base": {
            "model": AutoModelForSequenceClassification.from_pretrained(
                "textattack/bert-base-uncased-SST-2", num_labels=2
            ),
            "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        },
        "distilbert": {
            "model": AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english", num_labels=2
            ),
            "tokenizer": AutoTokenizer.from_pretrained("distilbert-base-uncased"),
        },
    }
    return models