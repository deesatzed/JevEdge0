"""Edge0 typed-decision scorer: option-logit readout, no generation.

One decision is one prefill.  After ``engine.prefill(ids)`` the base
engine holds the last chunk's logits (``engine/base.py``:
``next_logits()`` returns ``_last_logits``), which are the next-token
logits at the position where the answer letter belongs.  We select only
the declared option-letter token logits from that full-vocabulary row
and softmax across them.

No tokens are sampled and no text is generated, so the cost of a
decision is one prefill rather than a generation loop.

The resulting numbers are *conditional option scores*.  They are not
calibrated confidence and this module never calls them that.
"""

from __future__ import annotations

import time

from edge0.backends import core

from jevedge0.decision.prompt import (PROMPT_VERSION, DecisionError,
                                      encode_prompt, entropy, margin, softmax)

READOUT = ("last-position next-token logits restricted to declared "
           "answer-slot tokens")
PROBABILITY_STATUS = ("conditional option score; uncalibrated as decision "
                      "confidence")


class Edge0DecisionScorer:
    """Score typed decisions against an already-loaded Edge0 engine.

    The engine is shared with chat serving on purpose: Edge0 loads one
    model per process, so a separate decision service would mean a second
    copy of the weights.  Callers that also serve chat must hold the same
    lock around ``score`` that they hold around generation — Edge0's
    engine state (KV cache, prerouter cross-token state) is single-slot.
    """

    def __init__(self, engine, max_tokens: int = 8192):
        self.engine = engine
        self.max_tokens = max_tokens
        if engine._tok is None:
            raise DecisionError("engine has no tokenizer; cannot score")
        self.tokenizer = engine._tok

    # ---- single decision ---------------------------------------------

    def score(self, row: dict) -> dict:
        """Score one decision row and return its full record."""
        started = time.perf_counter()
        ids, slots, prompt_hash = encode_prompt(
            self.tokenizer, row, self.max_tokens)

        forward_start = time.perf_counter()
        logits = self._prefill_logits(ids)
        selected = self._select(logits, slots)
        forward_seconds = time.perf_counter() - forward_start

        probabilities = softmax(selected)
        option_ids = [opt["id"] for opt in row["options"]]
        best = max(range(len(probabilities)), key=probabilities.__getitem__)

        return {
            "id": row["id"],
            "choice": option_ids[best],
            "option_ids": option_ids,
            "probabilities": probabilities,
            "option_logits": selected,
            "margin": margin(probabilities),
            "entropy": entropy(probabilities),
            "input_tokens": len(ids),
            "forward_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - started,
            "prompt_sha256": prompt_hash,
            "prompt_version": PROMPT_VERSION,
            "model": self.model_metadata(),
            "method": "option_logit_readout",
            "readout": READOUT,
            "probability_status": PROBABILITY_STATUS,
        }

    def _prefill_logits(self, ids: list[int]):
        """Reset engine state, prefill, and return the last-position logits.

        The reset is mandatory, not hygiene: ``ChatSession.run`` resets for
        the same reason (``server/chat.py``) — leftover KV and prerouter
        cross-token state from a previous request corrupts the next one
        from its first token.
        """
        self.engine.reset()
        self.engine.prefill(ids)
        logits = self.engine.next_logits()
        if logits is None:
            raise DecisionError("engine produced no logits during prefill")
        return logits

    @staticmethod
    def _select(logits, slots: list[int]) -> list[float]:
        """Pull the option-slot logits out of the full-vocabulary row.

        Edge0 forwards return shape ``(batch, vocab)`` or ``(batch, seq,
        vocab)`` depending on the family path; reduce to the final
        position's vocabulary row before indexing the slots.
        """
        row = logits
        while getattr(row, "ndim", 1) > 1:
            row = row[-1]
        core.eval(row)
        values = [float(row[token].item()) for token in slots]
        if len(values) != len(slots):
            raise DecisionError("slot selection lost entries")
        return values

    def model_metadata(self) -> dict:
        engine = self.engine
        return {
            "name": getattr(engine, "name", "edge0"),
            "source": getattr(engine, "dir", None),
            "backend": "mlx",
        }

    # ---- several criteria against one state ---------------------------

    def score_batch(self, rows: list[dict]) -> list[dict]:
        """Score several decision rows.

        Edge0 keeps one mutable KV cache and does not implement JEV's
        shared-prefix branching, so each row is prefilled independently
        even when the rows share a state.  This is a real serialization
        cost, documented rather than hidden.
        """
        return [self.score(row) for row in rows]
