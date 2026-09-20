"""Local sentence embeddings: BERT inference on MLX.

Edge0's backend contract keeps the whole stack on MLX; pulling in torch
just to run a 22M-parameter encoder would put a second compute stack on
the machine for no benefit.  This module implements the BERT encoder
directly against ``mlx`` and loads the real
``sentence-transformers/all-MiniLM-L6-v2`` weights.

The pooling is mean-over-attention-mask followed by L2 normalization,
which is what the sentence-transformers configuration for this model
specifies (``1_Pooling/config.json``: mean tokens).  Getting that wrong
silently degrades every retrieval, so ``verify_parity`` checks the known
properties of this model's output space.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def resolve_model_path(model: str = DEFAULT_MODEL) -> str:
    """Find the model locally, downloading only if it is absent.

    A local directory path is used as-is.  Otherwise the HuggingFace
    cache is consulted; a download happens only when the weights are not
    already present.
    """
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(
        model, allow_patterns=["*.json", "*.txt", "model.safetensors"])


class BertEmbedder:
    """Mean-pooled BERT sentence encoder on MLX."""

    def __init__(self, model: str = DEFAULT_MODEL, max_length: int = 256):
        self.path = resolve_model_path(model)
        self.name = model
        with open(os.path.join(self.path, "config.json")) as fh:
            self.config = json.load(fh)
        self.max_length = min(
            max_length, self.config.get("max_position_embeddings", 512))
        self.hidden = self.config["hidden_size"]
        self.heads = self.config["num_attention_heads"]
        self.layers = self.config["num_hidden_layers"]
        self.eps = self.config.get("layer_norm_eps", 1e-12)
        self.head_dim = self.hidden // self.heads
        self._load_weights()
        self._load_tokenizer()

    def _load_weights(self) -> None:
        weights = mx.load(os.path.join(self.path, "model.safetensors"))
        # Checkpoints carry either bare or "bert."-prefixed names.
        self.w = {k.replace("bert.", "", 1): v for k, v in weights.items()}
        required = "embeddings.word_embeddings.weight"
        if required not in self.w:
            raise RuntimeError(
                f"{self.path}: missing {required}; not a BERT checkpoint")

    def _load_tokenizer(self) -> None:
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(
            os.path.join(self.path, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(
            pad_id=0, pad_token="[PAD]", length=None)

    # ---- model ---------------------------------------------------------

    def _layer_norm(self, x, prefix: str):
        weight = self.w[f"{prefix}.weight"]
        bias = self.w[f"{prefix}.bias"]
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + self.eps) * weight + bias

    def _linear(self, x, prefix: str):
        # HF stores Linear weights as (out, in); x @ W.T
        return x @ self.w[f"{prefix}.weight"].T + self.w[f"{prefix}.bias"]

    def _encode_batch(self, ids, mask):
        seq = ids.shape[1]
        positions = mx.arange(seq)[None, :]
        x = (self.w["embeddings.word_embeddings.weight"][ids]
             + self.w["embeddings.position_embeddings.weight"][positions]
             + self.w["embeddings.token_type_embeddings.weight"][
                 mx.zeros((1, seq), dtype=mx.int32)])
        x = self._layer_norm(x, "embeddings.LayerNorm")

        # Additive mask: padded positions pushed out of the softmax.
        attn_bias = (1.0 - mask[:, None, None, :].astype(mx.float32)) * -1e9

        for i in range(self.layers):
            p = f"encoder.layer.{i}"
            b, s, _ = x.shape

            def heads(t):
                return t.reshape(b, s, self.heads, self.head_dim
                                 ).transpose(0, 2, 1, 3)

            q = heads(self._linear(x, f"{p}.attention.self.query"))
            k = heads(self._linear(x, f"{p}.attention.self.key"))
            v = heads(self._linear(x, f"{p}.attention.self.value"))

            scores = (q @ k.transpose(0, 1, 3, 2)) / (self.head_dim ** 0.5)
            weights = mx.softmax(scores + attn_bias, axis=-1)
            context = (weights @ v).transpose(0, 2, 1, 3).reshape(b, s, -1)

            attn_out = self._linear(context, f"{p}.attention.output.dense")
            x = self._layer_norm(x + attn_out, f"{p}.attention.output.LayerNorm")

            intermediate = self._linear(x, f"{p}.intermediate.dense")
            # BERT uses exact gelu (erf form), not the tanh approximation.
            gelu = intermediate * 0.5 * (1.0 + mx.erf(intermediate / (2 ** 0.5)))
            layer_out = self._linear(gelu, f"{p}.output.dense")
            x = self._layer_norm(x + layer_out, f"{p}.output.LayerNorm")

        # Mean pooling over real tokens only.
        m = mask[:, :, None].astype(mx.float32)
        pooled = mx.sum(x * m, axis=1) / mx.maximum(mx.sum(m, axis=1), 1e-9)
        norm = mx.sqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True))
        return pooled / mx.maximum(norm, 1e-12)

    # ---- public API -----------------------------------------------------

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed texts; returns ``(n, hidden)`` L2-normalized float32."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.hidden), dtype=np.float32)
        out = []
        for start in range(0, len(texts), batch_size):
            batch = [t if t.strip() else "[UNK]"
                     for t in texts[start:start + batch_size]]
            encoded = self.tokenizer.encode_batch(batch)
            ids = mx.array([e.ids for e in encoded], dtype=mx.int32)
            mask = mx.array([e.attention_mask for e in encoded], dtype=mx.int32)
            vectors = self._encode_batch(ids, mask)
            mx.eval(vectors)
            out.append(np.array(vectors, dtype=np.float32))
        return np.concatenate(out, axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def verify_parity(self) -> dict:
        """Check the encoder behaves like a working sentence embedder.

        Verifies unit norm, that related sentences score above unrelated
        ones, and that batching does not change a vector.  A broken
        weight mapping or pooling shows up here rather than as quietly
        poor retrieval later.
        """
        probes = [
            "The emergency department is overcrowded tonight.",
            "The ED has too many patients waiting for beds.",
            "I baked a lemon cake for the office party.",
        ]
        vectors = self.encode(probes)
        norms = np.linalg.norm(vectors, axis=1)
        related = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        single = self.encode_one(probes[0])
        batch_drift = float(np.max(np.abs(single - vectors[0])))
        return {
            "dimension": int(vectors.shape[1]),
            "unit_norm": bool(np.allclose(norms, 1.0, atol=1e-4)),
            "related_similarity": related,
            "unrelated_similarity": unrelated,
            "separates_meaning": bool(related > unrelated + 0.15),
            "batch_consistency": batch_drift,
            "batch_stable": bool(batch_drift < 1e-4),
        }


_EMBEDDER: BertEmbedder | None = None


def get_embedder(model: str = DEFAULT_MODEL) -> BertEmbedder:
    """Process-wide embedder (loading the encoder repeatedly is waste)."""
    global _EMBEDDER
    if _EMBEDDER is None or _EMBEDDER.name != model:
        _EMBEDDER = BertEmbedder(model)
    return _EMBEDDER
