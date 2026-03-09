# ============================================================
#  AXIOM · neural_net.py
#  Feed-forward neural network — pure Python
#  Uses matrix.py for all math ops, no external libs
#  Architecture: input → Dense+ReLU → Dense+ReLU → Softmax
# ============================================================

import math
import json
import random

from matrix import (
    zeros, rand_matrix,
    matvec, outer,
    add, sub, mul_scalar, mul_elem,
    relu, relu_deriv,
    leaky_relu, leaky_relu_deriv,
    softmax,
    cross_entropy, cross_entropy_deriv,
    clip, clip_matrix,
    transpose,
)


class Layer:
    """Single fully-connected layer with biases."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        scale = math.sqrt(2.0 / in_dim)   # He init
        self.W  = rand_matrix(out_dim, in_dim, scale)
        self.b  = [0.0] * out_dim
        self.activation = activation

        # Adam optimizer state
        self.mW = zeros(out_dim, in_dim)
        self.mb = [0.0] * out_dim
        self.vW = zeros(out_dim, in_dim)
        self.vb = [0.0] * out_dim
        self.t  = 0   # time step

        # Cache for backprop
        self._input  = []
        self._pre_act = []
        self._output  = []

    # ── Forward ──────────────────────────────────────────────

    def forward(self, x: list) -> list:
        self._input = x[:]
        pre = [sum(self.W[i][j] * x[j] for j in range(len(x))) + self.b[i]
               for i in range(len(self.b))]
        self._pre_act = pre[:]

        if self.activation == "relu":
            out = relu(pre)
        elif self.activation == "leaky_relu":
            out = leaky_relu(pre)
        elif self.activation == "softmax":
            out = softmax(pre)
        else:
            out = pre[:]   # linear

        self._output = out[:]
        return out

    # ── Backward ─────────────────────────────────────────────

    def backward(self, grad_out: list) -> list:
        """
        grad_out : gradient w.r.t. this layer's output
        returns  : gradient w.r.t. this layer's input
        """
        if self.activation == "relu":
            delta = mul_elem(grad_out, relu_deriv(self._pre_act))
        elif self.activation == "leaky_relu":
            delta = mul_elem(grad_out, leaky_relu_deriv(self._pre_act))
        elif self.activation == "softmax":
            delta = grad_out[:]   # softmax grad already computed at loss
        else:
            delta = grad_out[:]

        # Gradients
        dW = outer(delta, self._input)   # (out_dim x in_dim)
        db = delta[:]

        # Gradient w.r.t. input (for previous layer)
        Wt = transpose(self.W)           # (in_dim x out_dim)
        dx = matvec(Wt, delta)

        return dx, dW, db

    # ── Adam update ──────────────────────────────────────────

    def update(self, dW: list, db: list,
               lr: float = 1e-3,
               beta1: float = 0.9, beta2: float = 0.999,
               eps: float = 1e-8) -> None:
        self.t += 1
        t = self.t

        # Weights
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                g = dW[i][j]
                self.mW[i][j] = beta1 * self.mW[i][j] + (1 - beta1) * g
                self.vW[i][j] = beta2 * self.vW[i][j] + (1 - beta2) * g * g
                m_hat = self.mW[i][j] / (1 - beta1 ** t)
                v_hat = self.vW[i][j] / (1 - beta2 ** t)
                self.W[i][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        # Biases
        for i in range(len(self.b)):
            g = db[i]
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * g
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * g * g
            m_hat = self.mb[i] / (1 - beta1 ** t)
            v_hat = self.vb[i] / (1 - beta2 ** t)
            self.b[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "W": self.W, "b": self.b,
            "activation": self.activation,
            "mW": self.mW, "mb": self.mb,
            "vW": self.vW, "vb": self.vb,
            "t": self.t,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layer":
        in_dim  = len(d["W"][0])
        out_dim = len(d["W"])
        obj = cls(in_dim, out_dim, d["activation"])
        obj.W  = d["W"];  obj.b  = d["b"]
        obj.mW = d["mW"]; obj.mb = d["mb"]
        obj.vW = d["vW"]; obj.vb = d["vb"]
        obj.t  = d["t"]
        return obj


class NeuralNet:
    """
    Feed-forward network with arbitrary hidden layers.
    Loss: sparse categorical cross-entropy + softmax output.
    Optimizer: Adam per layer.
    """

    def __init__(self, layer_sizes: list, lr: float = 1e-3):
        """
        layer_sizes: e.g. [input_dim, 128, 64, num_classes]
        """
        self.lr     = lr
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            is_last  = (i == len(layer_sizes) - 2)
            act      = "softmax" if is_last else "leaky_relu"
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], act))

    # ── Forward ──────────────────────────────────────────────

    def forward(self, x: list) -> list:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ── Train step ───────────────────────────────────────────

    def train_step(self, x: list, y_idx: int) -> float:
        # Forward
        probs = self.forward(x)
        loss  = cross_entropy(probs, y_idx)

        # Backward — combined softmax + CE gradient at output
        grad = cross_entropy_deriv(probs, y_idx)

        for layer in reversed(self.layers):
            grad, dW, db = layer.backward(grad)
            dW = clip_matrix(dW)
            db = clip(db)
            layer.update(dW, db, lr=self.lr)

        return loss

    # ── Predict ──────────────────────────────────────────────

    def predict(self, x: list) -> tuple:
        """Returns (class_index, confidence, all_probs)."""
        probs    = self.forward(x)
        best_idx = probs.index(max(probs))
        return best_idx, probs[best_idx], probs

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "lr":     self.lr,
            "layers": [l.to_dict() for l in self.layers],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "NeuralNet":
        with open(path) as f:
            d = json.load(f)
        obj = cls.__new__(cls)
        obj.lr     = d["lr"]
        obj.layers = [Layer.from_dict(ld) for ld in d["layers"]]
        return obj
