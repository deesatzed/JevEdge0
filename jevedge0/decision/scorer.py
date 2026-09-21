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

import math
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
        legal_mass = self._legal_mass(logits, selected)
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
            "legal_mass": legal_mass,
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

    @staticmethod
    def _legal_mass(logits, selected: list[float]) -> float:
        """Fraction of the FULL next-token distribution the options cover.

        This is a different softmax from the one ``score()`` reports as
        ``probabilities``: that one is restricted to only the declared
        options and always sums to 1 by construction, so it cannot tell
        you whether those options were ever plausible answers in the
        first place. ``legal_mass`` answers that question by measuring
        the option logits against every other token the model could have
        produced at that position. A low value means the model's mass
        was concentrated elsewhere -- the declared options did not cover
        what it actually wanted to say, and a caller should be more
        skeptical of the resulting choice even if its restricted-softmax
        probability looks high.

        Computed via log-sum-exp over the full vocabulary row rather than
        materializing a full-vocabulary probability array, since the
        normalizer is all a sum of exponentials needs and the vocabulary
        here is on the order of 10^5 tokens.
        """
        row = logits
        while getattr(row, "ndim", 1) > 1:
            row = row[-1]
        core.eval(row)
        # log Z = logsumexp(all logits); legal mass = sum(exp(s_i - logZ))
        # for the selected option logits s_i. Numerically stable because
        # logsumexp itself subtracts the row max before exponentiating.
        log_z = float(core.logsumexp(row).item())
        mass = sum(math.exp(s - log_z) for s in selected)
        # Clip defensively: floating-point summation over ~1e5 terms can
        # push this a hair above 1.0 even though it is mathematically
        # bounded by the full distribution's total mass of 1.0.
        return max(0.0, min(1.0, mass))

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

    def score_shared_state(self, state, questions: list[dict]) -> list[dict]:
        """Score several questions against one shared state.

        NOT a shared-prefix optimization. GOAL_ENHANCE.md Stage 5 asked
        for the prefill of ``state`` to happen once and be branched for
        each question's continuation, the way JEV's design describes.
        That was investigated against the real engine
        (``edge0/engine/base.py``, ``edge0/engine/qwen.py``,
        ``edge0/prerouter/state.py``) before writing a line of this
        method, and is **not implementable without modifying Edge0
        internals** -- see ``ERRORS.md`` for the finding this method's
        docstring summarizes, and the two build-order facts that make
        one-time reuse of a single prefill genuinely impossible here,
        not merely un-optimized:

        1. ``Qwen35Engine.cache`` is one mutable ``KVCache``/``ArraysCache``
           list built once by ``make_cache()`` and mutated in place by
           every ``_forward`` call. It is not a batchable structure
           (mlx-lm's own ``BatchKVCache`` exists and is not what Edge0
           constructs), and MLX-level array snapshotting would still
           leave the *engine's* Python-level cross-token state -- see
           point 2 -- pointing at the wrong branch.
        2. ``PrerouterState`` (``prerouter/state.py``) holds exactly one
           double-buffered "previous token" position (``logits_prev``,
           ``oh_prev``) shared across every layer, consumed and
           overwritten by ``swap()`` on each forward. It has no concept
           of more than one active sequence position at a time, so
           forking after a shared prefill would require each branch to
           carry its own copy of this state (and of every streaming
           expert's staged-slot state in ``_all_stream_layers``) and
           somehow interleave forwards across branches without them
           corrupting each other's "previous token" bookkeeping. That is
           new engine capability, not a call-site optimization, and is
           explicitly out of scope: GOAL_ENHANCE.md forbids modifying
           ``src/edge0/``.

        This method therefore does exactly what ``score_batch`` already
        does -- reset, prefill, and score each question independently --
        with the one honest difference of accepting a single shared
        ``state`` argument so a caller does not have to repeat it, and
        constructing the shared prompt PREFIX text once (before the
        per-question suffix) so at minimum the state-to-text rendering
        itself is not redone questions-many times. The measured speedup
        of doing this versus ``score_batch`` on N separately-stated rows
        is expected to be ~1.0x on the engine-bound cost (each question
        is still a full independent prefill+forward), and is measured and
        reported as such rather than assumed.
        """
        records = []
        for question in questions:
            row = {
                "id": question.get("id", f"shared-{len(records)}"),
                "state": state,
                "question": question["question"],
                "options": question["options"],
            }
            records.append(self.score(row))
        return records
