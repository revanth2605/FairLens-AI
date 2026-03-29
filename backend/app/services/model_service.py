"""
FairLens AI — Model Inference Service
Loads the saved DistilBERT model and predicts bias in text
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL_DIR = "revanthkothamasu26/fairlens-model"
MAX_LENGTH = 128
LABELS     = {0: "Not Biased", 1: "Biased"}

# ── Load model once at startup ─────────────────────────────────────────────
# We load it globally so it doesn't reload on every request (slow)
print("Loading model from saved_model/...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()   # set to evaluation mode (disables dropout etc.)
print("Model ready!")


def predict(text: str) -> dict:
    """
    Takes a raw string, returns prediction + confidence scores.

    Example output:
    {
        "label": "Biased",
        "confidence": 0.91,
        "scores": {"Not Biased": 0.09, "Biased": 0.91}
    }
    """
    # Step 1: Clean and tokenize the input text
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"     # PyTorch tensors
    )

    # Step 2: Run through the model (no gradient needed for inference)
    with torch.no_grad():
        outputs = model(**inputs)

    # Step 3: Convert raw logits → probabilities using softmax
    # logits example: [-0.5, 1.8] → probabilities: [0.09, 0.91]
    probs = torch.softmax(outputs.logits, dim=1).squeeze().numpy()

    # Step 4: Pick the highest probability as the prediction
    predicted_class = int(np.argmax(probs))

    return {
        "label":      LABELS[predicted_class],
        "confidence": round(float(probs[predicted_class]), 4),
        "scores": {
            "Not Biased": round(float(probs[0]), 4),
            "Biased":     round(float(probs[1]), 4),
        }
    }