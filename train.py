# ============================================================
#  AXIOM · train.py
#  Trains the NLP model and saves artifacts
#  Run: python train.py
#  Outputs: model.json   vectorizer.json   labels.json
# ============================================================

import json
import random
import math

from training_data import training_data
from tfidf        import TFIDFVectorizer
from neural_net   import NeuralNet

# ── Config ───────────────────────────────────────────────────
EPOCHS       = 300
LR           = 5e-3
HIDDEN_SIZES = [256, 128]
MAX_FEATURES = 1500

# ── 1. Prepare ───────────────────────────────────────────────
texts  = [t[0] for t in training_data]
labels = [t[1] for t in training_data]

# Label encoding
unique_labels = sorted(set(labels))
label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
idx_to_label  = {i: l for l, i in label_to_idx.items()}

print(f"Intents  : {unique_labels}")
print(f"Samples  : {len(texts)}")

# Vectorize
vectorizer = TFIDFVectorizer(max_features=MAX_FEATURES)
vectorizer.fit(texts)
X = [vectorizer.transform(t) for t in texts]
y = [label_to_idx[l] for l in labels]

input_dim   = len(X[0])
num_classes = len(unique_labels)
layer_sizes = [input_dim] + HIDDEN_SIZES + [num_classes]

print(f"Input dim: {input_dim}  |  Layers: {layer_sizes}")

# ── 2. Train ─────────────────────────────────────────────────
net = NeuralNet(layer_sizes, lr=LR)

pairs = list(zip(X, y))

best_loss  = float('inf')
best_state = None
patience   = 40
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    random.shuffle(pairs)
    total_loss = 0.0
    correct    = 0

    for x, yi in pairs:
        loss = net.train_step(x, yi)
        total_loss += loss
        pred, _, _ = net.predict(x)
        if pred == yi:
            correct += 1

    avg_loss = total_loss / len(pairs)
    acc      = correct / len(pairs) * 100

    if avg_loss < best_loss - 1e-5:
        best_loss  = avg_loss
        best_state = net.to_dict()
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}  loss={avg_loss:.4f}  acc={acc:.1f}%")

    if no_improve >= patience:
        print(f"Early stop at epoch {epoch}")
        break

# ── 3. Save best model ───────────────────────────────────────
# Restore best weights
import copy
net_best = NeuralNet.__new__(NeuralNet)
from neural_net import Layer
net_best.lr     = best_state["lr"]
net_best.layers = [Layer.from_dict(ld) for ld in best_state["layers"]]

net_best.save("model.json")
vectorizer.save("vectorizer.json")

with open("labels.json", "w") as f:
    json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f)

print(f"\nDone! Best loss: {best_loss:.4f}")
print("Saved: model.json  vectorizer.json  labels.json")

# ── 4. Quick sanity check ────────────────────────────────────
print("\n── Sanity check ──")
tests = [
    "hey",
    "who are you",
    "who made you",
    "what can you do",
    "bye",
    "what is the weather",
]
for t in tests:
    vec            = vectorizer.transform(t)
    idx, conf, _   = net_best.predict(vec)
    label          = idx_to_label[str(idx)]
    print(f"  '{t}' → {label} ({conf*100:.1f}%)")
