"""
prediction.py  -  Use a trained model on a new complaint (deep learning part)
Also contains simple word-importance (explainability).
"""
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text, MAX_LEN, MODEL_DIR, OUTPUT_DIR


def load_resources(model_name):
    """Load the trained model, the tokenizer and the category names."""
    model_path = os.path.join(MODEL_DIR, model_name.lower() + "_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Run preprocess.py, train.py, compare_models.py.")
    model = load_model(model_path)
    with open(os.path.join(MODEL_DIR, "tokenizer.json")) as f:
        tokenizer = tokenizer_from_json(f.read())
    with open(os.path.join(MODEL_DIR, "classes.json")) as f:
        class_names = json.load(f)
    return model, tokenizer, class_names


def text_to_padded(texts, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_LEN, padding="pre", truncating="post")


def predict_category(complaint, model, tokenizer, class_names):
    """Returns (category, confidence, cleaned_text, all_probabilities)."""
    cleaned = clean_text(complaint)
    if len(cleaned.split()) == 0:
        raise ValueError("Complaint has no readable words.")
    padded = text_to_padded([cleaned], tokenizer)
    probabilities = model.predict(padded, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    return class_names[best_index], float(probabilities[best_index]), cleaned, probabilities


def word_importance(cleaned_text, model, tokenizer, class_index, max_words=40):
    """
    Simple 'leave one word out' (occlusion) method:
    remove one word at a time and see how much the probability of the predicted
    category drops. A bigger drop = the word mattered more.
    This is an APPROXIMATE indicator, not an exact causal explanation.
    """
    words = cleaned_text.split()[:max_words]
    original = model.predict(text_to_padded([" ".join(words)], tokenizer), verbose=0)[0][class_index]

    # Build all "one word removed" versions and predict them in one batch
    versions = []
    for i in range(len(words)):
        versions.append(" ".join(words[:i] + words[i + 1:]))
    new_probabilities = model.predict(text_to_padded(versions, tokenizer), verbose=0)[:, class_index]

    importance = []
    for i in range(len(words)):
        importance.append((words[i], float(original - new_probabilities[i])))
    return importance
