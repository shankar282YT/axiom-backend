# ============================================================
#  AXIOM · tfidf.py
#  TF-IDF vectorizer — pure Python, no external libs
#  Character n-gram based (robust to typos / short inputs)
# ============================================================

import math
import json
import re


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _char_ngrams(text: str, n_min: int = 2, n_max: int = 4) -> list:
    """Extract character n-grams with word boundaries (#word#)."""
    tokens = text.split()
    grams = []
    for token in tokens:
        padded = f"#{token}#"
        for n in range(n_min, n_max + 1):
            for i in range(len(padded) - n + 1):
                grams.append(padded[i:i + n])
    return grams


class TFIDFVectorizer:
    """
    Fit on a list of strings, transform to TF-IDF float vectors.
    Serializable to/from plain dict (for saving alongside model).
    """

    def __init__(self, max_features: int = 2000, n_min: int = 2, n_max: int = 4):
        self.max_features = max_features
        self.n_min        = n_min
        self.n_max        = n_max
        self.vocab: dict  = {}        # gram → index
        self.idf:  list   = []        # idf per vocab entry
        self.fitted       = False

    # ── Fit ──────────────────────────────────────────────────

    def fit(self, texts: list) -> "TFIDFVectorizer":
        normalized = [_normalize(t) for t in texts]
        N = len(normalized)

        # Document frequency count
        df: dict = {}
        for text in normalized:
            grams = set(_char_ngrams(text, self.n_min, self.n_max))
            for g in grams:
                df[g] = df.get(g, 0) + 1

        # Keep top-k by df
        sorted_grams = sorted(df.items(), key=lambda x: -x[1])
        top_grams    = [g for g, _ in sorted_grams[:self.max_features]]

        self.vocab = {g: i for i, g in enumerate(top_grams)}
        self.idf   = [
            math.log((N + 1) / (df.get(g, 0) + 1)) + 1.0
            for g in top_grams
        ]
        self.fitted = True
        return self

    # ── Transform ────────────────────────────────────────────

    def transform(self, text: str) -> list:
        """Return a TF-IDF float vector for a single string."""
        assert self.fitted, "Call fit() first."
        normalized = _normalize(text)
        grams      = _char_ngrams(normalized, self.n_min, self.n_max)
        total      = len(grams) or 1

        # Term frequency
        tf: dict = {}
        for g in grams:
            tf[g] = tf.get(g, 0) + 1

        vec = [0.0] * len(self.vocab)
        for g, idx in self.vocab.items():
            if g in tf:
                vec[idx] = (tf[g] / total) * self.idf[idx]

        # L2 normalize
        magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / magnitude for v in vec]

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "max_features": self.max_features,
            "n_min":        self.n_min,
            "n_max":        self.n_max,
            "vocab":        self.vocab,
            "idf":          self.idf,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TFIDFVectorizer":
        obj = cls(d["max_features"], d["n_min"], d["n_max"])
        obj.vocab  = d["vocab"]
        obj.idf    = d["idf"]
        obj.fitted = True
        return obj

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "TFIDFVectorizer":
        with open(path) as f:
            return cls.from_dict(json.load(f))
