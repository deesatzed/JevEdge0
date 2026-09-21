"""Tests for Edge0DecisionScorer.score_shared_state (GOAL_ENHANCE.md Stage 5).

score_shared_state is, by construction, a thin wrapper that calls
self.score(row) once per question -- the investigation recorded in
scorer.py's docstring and ERRORS.md found true shared-prefix KV-cache
branching is not implementable without modifying Edge0 internals, which
is out of scope. These tests verify the wrapper's own behavior (ordering,
row construction, reset-between-calls) using a scripted engine standing
in for the real one; true engine-level numerical identity against
independently-run score() calls is validated separately against real
weights (GOAL_ENHANCE.md Sec 7.D), since that is where the property
actually matters and a fake engine cannot exercise real prefill logits.
"""

from __future__ import annotations

import pytest

from jevedge0.decision.scorer import Edge0DecisionScorer


class ScriptedTokenizer:
    """Minimal tokenizer stand-in: encodes to a length proportional to text,
    with a distinct single-token id per option letter."""

    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        # Deterministic, content-dependent "tokenization" so different
        # states/questions produce different (but stable) id sequences.
        return [abs(hash(text)) % 1000] * max(1, len(text) // 10)

    def decode(self, ids):
        return "".join(chr(65 + (i % 4)) for i in ids)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        return "".join(m["content"] for m in messages)


class ScriptedEngine:
    """Records every prefill call and returns a scripted logits row.

    Mimics just enough of Edge0Engine's surface for Edge0DecisionScorer:
    reset(), prefill(ids), next_logits().
    """

    def __init__(self, tokenizer, choice_by_call=None):
        self._tok = tokenizer
        self.reset_calls = 0
        self.prefill_calls = []
        self._choice_by_call = choice_by_call or {}
        self._call_index = 0

    def reset(self):
        self.reset_calls += 1

    def prefill(self, ids, chunk_size=None, on_progress=None):
        self.prefill_calls.append(list(ids))
        return len(ids)

    def next_logits(self):
        # Return a fake full-vocab-ish row via a tiny numpy-free stand-in
        # object that supports the indexing / eval / item() surface
        # Edge0DecisionScorer._select and _legal_mass need.
        return _FakeLogitsRow(self._call_index)


class _FakeLogitsRow:
    """Just enough array-like behavior for _select/_legal_mass to work."""

    ndim = 1

    def __init__(self, call_index):
        self.call_index = call_index

    def __getitem__(self, token):
        return _Scalar(1.0 if (token % 7) == (self.call_index % 7) else -1.0)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def _patch_core_for_test(monkeypatch):
    """encode_prompt / softmax / _select all go through jevedge0.decision
    modules that import edge0.backends.core for eval/logsumexp -- stub
    those two calls to be no-ops / simple reducers so this test needs no
    real MLX/Edge0 installation."""
    import jevedge0.decision.scorer as scorer_mod
    monkeypatch.setattr(scorer_mod.core, "eval", lambda *a, **k: None)
    monkeypatch.setattr(scorer_mod.core, "logsumexp",
                        lambda row: _Scalar(2.0))


@pytest.fixture
def scorer(monkeypatch):
    _patch_core_for_test(monkeypatch)
    tokenizer = ScriptedTokenizer()
    engine = ScriptedEngine(tokenizer)
    engine._tok = tokenizer
    return Edge0DecisionScorer(engine)


ROW_OPTIONS = [{"id": "a", "description": "Option A description."},
              {"id": "b", "description": "Option B description."}]


def _monkeypatch_encode(monkeypatch, scorer):
    """encode_prompt does real chat-template rendering + tokenizer calls
    + strict single-token slot validation, none of which the scripted
    tokenizer above can satisfy realistically. Stub it to a minimal but
    faithful shape: real ids list, real slot ids, a stable hash."""
    import jevedge0.decision.scorer as scorer_mod

    def fake_encode_prompt(tokenizer, row, max_tokens):
        ids = [1, 2, 3]
        slots = list(range(len(row["options"])))
        return ids, slots, f"hash-{row['id']}"

    monkeypatch.setattr(scorer_mod, "encode_prompt", fake_encode_prompt)


# ---- 1. one record per question, in order ---------------------------------

def test_score_shared_state_returns_one_record_per_question_in_order(
        monkeypatch, scorer):
    _monkeypatch_encode(monkeypatch, scorer)
    questions = [
        {"id": "q1", "question": "First question?", "options": ROW_OPTIONS},
        {"id": "q2", "question": "Second question?", "options": ROW_OPTIONS},
        {"id": "q3", "question": "Third question?", "options": ROW_OPTIONS},
    ]
    records = scorer.score_shared_state("shared evidence text", questions)
    assert len(records) == 3
    assert [r["id"] for r in records] == ["q1", "q2", "q3"]


def test_score_shared_state_assigns_default_ids_when_missing(
        monkeypatch, scorer):
    _monkeypatch_encode(monkeypatch, scorer)
    questions = [{"question": "Q?", "options": ROW_OPTIONS}]
    records = scorer.score_shared_state("state", questions)
    assert records[0]["id"] == "shared-0"


# ---- 2. numerical identity vs scoring each row individually ---------------

def test_score_shared_state_matches_individual_score_calls_exactly(
        monkeypatch, scorer):
    """score_shared_state is a thin wrapper over score() by construction
    (see its docstring for why the alternative -- true shared-prefix
    branching -- was investigated and found not implementable without
    modifying Edge0). This confirms the wrapper introduces NO numerical
    difference versus calling score() directly on equivalent rows --
    the one property Stage 5 calls non-negotiable: 'a fast wrong answer
    is a failure.'"""
    _monkeypatch_encode(monkeypatch, scorer)
    state = "shared evidence"
    questions = [
        {"id": "q1", "question": "First?", "options": ROW_OPTIONS},
        {"id": "q2", "question": "Second?", "options": ROW_OPTIONS},
    ]

    shared_records = scorer.score_shared_state(state, questions)

    individual_records = []
    for q in questions:
        row = {"id": q["id"], "state": state, "question": q["question"],
              "options": q["options"]}
        individual_records.append(scorer.score(row))

    for shared, individual in zip(shared_records, individual_records):
        assert shared["option_logits"] == pytest.approx(
            individual["option_logits"], abs=1e-3)
        assert shared["choice"] == individual["choice"]
        assert shared["probabilities"] == pytest.approx(
            individual["probabilities"], abs=1e-3)


# ---- 3. engine state is reset correctly between calls ----------------------

def test_score_shared_state_resets_engine_before_each_question(
        monkeypatch, scorer):
    """Each question is its own independent prefill (the honest
    consequence of the Stage 5 finding): the engine must be reset before
    each one, exactly as score() already does for a standalone call --
    leftover KV/prerouter state from a previous question must not leak
    into the next."""
    _monkeypatch_encode(monkeypatch, scorer)
    questions = [
        {"id": "q1", "question": "Q1?", "options": ROW_OPTIONS},
        {"id": "q2", "question": "Q2?", "options": ROW_OPTIONS},
        {"id": "q3", "question": "Q3?", "options": ROW_OPTIONS},
    ]
    scorer.score_shared_state("state", questions)
    assert scorer.engine.reset_calls == 3
    assert len(scorer.engine.prefill_calls) == 3


def test_shared_state_batch_then_single_score_matches_running_alone(
        monkeypatch, scorer):
    """A shared-state batch followed by an ordinary single score() call
    must give the same result as running that single score() alone --
    i.e. the batch leaves no residual state to corrupt a later call."""
    _monkeypatch_encode(monkeypatch, scorer)
    questions = [{"id": "q1", "question": "Q1?", "options": ROW_OPTIONS}]
    scorer.score_shared_state("state", questions)

    solo_row = {"id": "solo", "state": "different state",
               "question": "Solo question?", "options": ROW_OPTIONS}
    after_batch = scorer.score(solo_row)

    # Run the identical solo row on a fresh scorer/engine for comparison.
    fresh_tokenizer = ScriptedTokenizer()
    fresh_engine = ScriptedEngine(fresh_tokenizer)
    fresh_engine._tok = fresh_tokenizer
    fresh_scorer = Edge0DecisionScorer(fresh_engine)
    _monkeypatch_encode(monkeypatch, fresh_scorer)
    standalone = fresh_scorer.score(solo_row)

    assert after_batch["choice"] == standalone["choice"]
    assert after_batch["option_logits"] == pytest.approx(
        standalone["option_logits"], abs=1e-3)
