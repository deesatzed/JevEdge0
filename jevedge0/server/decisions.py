"""Decision endpoints served from the same process as Edge0 chat.

``/v1/decisions``       one criterion against one state
``/v1/decision-batch``  several criteria against one shared state

Both run on the engine that is already serving ``/v1/chat/completions``.
That sharing is the point: Edge0 loads one model per process, so a
separate decision service would mean a second copy of the weights in
memory.  The cost is that decisions and chat contend for the same single
engine slot — requests serialize on ``QueueServer._lock``.
"""

from __future__ import annotations

import threading
import time

from jevedge0.decision.prompt import DecisionError
from jevedge0.decision.scorer import Edge0DecisionScorer
from jevedge0.decision.stability import score_with_stability


def engine_lock(queue_server) -> threading.RLock:
    """Return the reentrant lock guarding this server's engine.

    Edge0's ``QueueServer`` guards its single engine slot with a plain
    ``threading.Lock``.  That is correct for Edge0, where one request is
    one generation, but it self-deadlocks here: the comparator holds the
    engine for a whole dual-channel run and calls ``QueueServer.chat``
    *inside* that run, which tries to take the same non-reentrant lock on
    the same thread and blocks forever.

    Promoting the lock to an ``RLock`` keeps the mutual exclusion that
    matters — two different threads still cannot touch the engine at once
    — while letting one thread hold the engine across several operations
    that are meant to be one atomic unit.  The swap happens once, and
    only if the server is still carrying a plain lock.
    """
    lock = queue_server._lock
    if not isinstance(lock, type(threading.RLock())):
        lock = threading.RLock()
        queue_server._lock = lock
    return lock


class DecisionService:
    """Bind a scorer to a QueueServer and answer decision payloads."""

    def __init__(self, queue_server, max_tokens: int = 8192):
        self.queue_server = queue_server
        self.lock = engine_lock(queue_server)
        self.scorer = Edge0DecisionScorer(queue_server.engine,
                                          max_tokens=max_tokens)

    # ---- payload parsing ---------------------------------------------

    @staticmethod
    def _row_from_payload(payload: dict, index: int = 0) -> dict:
        """Build a decision row from a request body.

        Accepts the ``jevadd.md`` shape (``state`` / ``criterion`` /
        ``options``) and also ``question`` as a synonym for ``criterion``
        so JEV-CPU JSONL rows can be posted unchanged.
        """
        if not isinstance(payload, dict):
            raise DecisionError("request body must be a JSON object")
        criterion = payload.get("criterion", payload.get("question"))
        if criterion is None:
            raise DecisionError("missing 'criterion' (or 'question')")
        row = {
            "id": str(payload.get("id") or f"decision-{index}"),
            "state": payload.get("state"),
            "question": criterion,
            "options": payload.get("options"),
        }
        if row["state"] is None:
            raise DecisionError("missing 'state'")
        if row["options"] is None:
            raise DecisionError("missing 'options'")
        return row

    @staticmethod
    def _public(record: dict) -> dict:
        """Shape one scorer record for the wire.

        ``probabilities`` goes out as an id->value object (the
        ``jevadd.md`` response shape) rather than a bare list, so a
        consumer cannot silently mis-pair values with options.
        """
        out = dict(record)
        out["probabilities"] = dict(
            zip(record["option_ids"], record["probabilities"]))
        out["option_logits"] = dict(
            zip(record["option_ids"], record["option_logits"]))
        if "stability" in record:
            stability = dict(record["stability"])
            stability.pop("trial_records", None)
            out["stability"] = stability
        return out

    # ---- endpoints ----------------------------------------------------

    def decide(self, payload: dict) -> dict:
        """POST /v1/decisions"""
        row = self._row_from_payload(payload)
        trials = int(payload.get("stability_trials", 1) or 1)
        started = time.time()
        with self.lock:
            if trials > 1:
                record = score_with_stability(
                    self.scorer, row, trials=trials,
                    seed=payload.get("seed", 0))
            else:
                record = self.scorer.score(row)
        result = self._public(record)
        result["object"] = "decision"
        result["created"] = int(started)
        return result

    def decide_batch(self, payload: dict) -> dict:
        """POST /v1/decision-batch

        One ``state`` with several ``criteria``; each criterion may bring
        its own options or inherit a shared ``options`` list.
        """
        if not isinstance(payload, dict):
            raise DecisionError("request body must be a JSON object")
        state = payload.get("state")
        if state is None:
            raise DecisionError("missing 'state'")
        criteria = payload.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise DecisionError("'criteria' must be a nonempty array")
        shared_options = payload.get("options")
        trials = int(payload.get("stability_trials", 1) or 1)
        seed = payload.get("seed", 0)

        rows = []
        for index, item in enumerate(criteria):
            if isinstance(item, str):
                item = {"criterion": item}
            if not isinstance(item, dict):
                raise DecisionError(
                    "each criterion must be a string or an object")
            merged = {
                "id": item.get("id") or f"criterion-{index}",
                "state": item.get("state", state),
                "criterion": item.get("criterion", item.get("question")),
                "options": item.get("options", shared_options),
            }
            rows.append(self._row_from_payload(merged, index))

        started = time.time()
        results = []
        # One lock acquisition for the whole batch: interleaving chat
        # generations between criteria would reset engine state between
        # rows that are meant to be read together.
        with self.lock:
            for row in rows:
                if trials > 1:
                    record = score_with_stability(
                        self.scorer, row, trials=trials, seed=seed)
                else:
                    record = self.scorer.score(row)
                results.append(self._public(record))
        return {
            "object": "decision.batch",
            "created": int(started),
            "model": self.queue_server.model_name,
            "results": results,
        }


def build_decision_handlers(queue_server, service=None) -> dict:
    """Return route handlers to merge into Edge0's dispatch dict."""
    service = service or DecisionService(queue_server)

    def guard(fn):
        def wrapped(payload: dict):
            try:
                return fn(payload)
            except DecisionError as exc:
                return {"error": {"message": str(exc),
                                  "type": "invalid_request"}}
        return wrapped

    return {
        "POST /v1/decisions": guard(service.decide),
        "POST /v1/decision-batch": guard(service.decide_batch),
    }
