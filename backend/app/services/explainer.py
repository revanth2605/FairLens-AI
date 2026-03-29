"""
FairLens AI — Explainability Service
Uses LIME to highlight which words caused the bias prediction
"""

import re
import string
import numpy as np
from lime.lime_text import LimeTextExplainer
from app.services.model_service import predict, tokenizer, model
import torch


# ── LIME needs a function that takes a LIST of texts and returns probabilities
# This is different from our single-text predict() function

def predict_proba(texts):
    """
    Takes a list of strings.
    Returns a 2D numpy array of shape (num_texts, 2)
    Each row = [probability_not_biased, probability_biased]

    LIME calls this function many times with slightly modified versions
    of your text (words randomly removed) to see what changes.
    """
    results = []

    for text in texts:
        # Tokenize the text
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        # Run through model
        with torch.no_grad():
            outputs = model(**inputs)

        # Convert logits → probabilities
        probs = torch.softmax(outputs.logits, dim=1).squeeze().numpy()
        results.append(probs)

    return np.array(results)   # shape: (num_texts, 2)


# ── Create the LIME explainer (once, globally)
explainer = LimeTextExplainer(
    class_names=["Not Biased", "Biased"],  # our two classes
    split_expression=r'\W+',               # split text into words
    bow=False,                             # don't use bag-of-words mode
    random_state=42
)


def explain(text: str, num_features: int = 10) -> dict:
    """
    Takes a sentence, returns:
    - The prediction (Biased / Not Biased)
    - Confidence score
    - Word importance scores (which words pushed toward bias)

    num_features = how many words to highlight (top N most important)
    """

    # Step 1: Get the main prediction first
    prediction = predict(text)

    # Step 2: Run LIME — it will call predict_proba ~500 times
    # with different versions of the text (words masked out)
    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,    # top N words to explain
        num_samples=300,              # how many variations to try
                                      # (lower = faster, less accurate)
                                      # (higher = slower, more accurate)
                                      # 300 is good for CPU
        labels=[1]                    # explain the "Biased" class
    )

    # Step 3: Extract word importance scores
    # Each item is (word, importance_score)
    # Positive score = pushed toward BIASED
    # Negative score = pushed toward NOT BIASED
    word_scores = explanation.as_list(label=1)

    # Step 4: Format into a clean dict
    words = []
    for word, score in word_scores:
        words.append({
            "word":      word,
            "score":     round(float(score), 4),
            "direction": "biased" if score > 0 else "not_biased"
        })

    # Sort by absolute importance (most important first)
    words.sort(key=lambda x: abs(x["score"]), reverse=True)

    return {
        "text":       text,
        "label":      prediction["label"],
        "confidence": prediction["confidence"],
        "scores":     prediction["scores"],
        "explanation": words       # list of {word, score, direction}
    }


def highlight_text(text: str) -> str:
    """
    Returns the text with HTML highlighting.
    Biased words → red background
    Neutral words → green background
    Intensity based on score strength.

    Use this to render colored text in the frontend.
    """
    result = explain(text)
    word_scores = {item["word"].lower(): item["score"]
                   for item in result["explanation"]}

    # Split text into words (keep original casing)
    tokens = text.split()
    highlighted = []

    for token in tokens:
        # Strip punctuation for lookup
        clean = token.lower().strip(string.punctuation)
        score = word_scores.get(clean, 0)

        if abs(score) < 0.01:
            # Not important — show as plain text
            highlighted.append(f'<span>{token}</span>')
        elif score > 0:
            # Biased word — red, intensity based on score
            intensity = min(int(abs(score) * 800), 255)
            highlighted.append(
                f'<span style="background:rgba(220,50,50,{min(abs(score)*3, 0.7):.2f});'
                f'border-radius:3px;padding:1px 3px;">{token}</span>'
            )
        else:
            # Counter-bias word — green
            highlighted.append(
                f'<span style="background:rgba(50,180,100,{min(abs(score)*3, 0.7):.2f});'
                f'border-radius:3px;padding:1px 3px;">{token}</span>'
            )

    return " ".join(highlighted)