# ============================================================
#  AXIOM · nlp.py
#  Inference wrapper — loads trained artifacts, predicts intent
# ============================================================

import json
import os

from tfidf      import TFIDFVectorizer
from neural_net import NeuralNet

_MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model.json")
_VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.json")
_LABELS_PATH     = os.path.join(os.path.dirname(__file__), "labels.json")

# Lazy-loaded singletons
_vectorizer:  TFIDFVectorizer | None = None
_model:       NeuralNet        | None = None
_idx_to_label: dict            | None = None


def _load():
    global _vectorizer, _model, _idx_to_label

    if _model is not None:
        return  # already loaded

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            "model.json not found — run `python train.py` first."
        )

    _vectorizer   = TFIDFVectorizer.load(_VECTORIZER_PATH)
    _model        = NeuralNet.load(_MODEL_PATH)

    with open(_LABELS_PATH) as f:
        labels_data  = json.load(f)
    _idx_to_label = {int(k): v for k, v in labels_data["idx_to_label"].items()}


def predict(text: str) -> dict:
    """
    Returns:
    {
        "intent":     str,
        "confidence": float,
        "all_probs":  { intent_name: probability, ... }
    }
    """
    _load()

    vec              = _vectorizer.transform(text)
    idx, conf, probs = _model.predict(vec)
    intent           = _idx_to_label[idx]

    all_probs = {
        _idx_to_label[i]: round(probs[i], 4)
        for i in range(len(probs))
    }

    return {
        "intent":     intent,
        "confidence": round(conf, 4),
        "all_probs":  all_probs,
    }
