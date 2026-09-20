"""JevEdge0 tests that require the real checkpoint and real weights.

Marked ``slow`` — Edge0's pytest config runs ``-m 'not slow'`` by default.
Run them with::

    pytest tests/test_jevedge0_slow.py -m slow

These use the real model and real embedding weights. Nothing here is
mocked; a missing checkpoint skips rather than substituting a fake.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.slow

MODEL_DIR = (os.environ.get("EDGE0_35B_MODEL")
             or ("models/edge0-35b" if os.path.isdir("models/edge0-35b")
                 else None))

ROW = {
    "id": "ed-strain",
    "state": ("Current ED census is 71. There are 14 admitted patients "
              "boarding, five patients awaiting ICU beds, eight patients in "
              "the waiting room, and predicted arrivals exceed available "
              "treatment spaces for the next four hours. Nursing staffing is "
              "two positions below plan."),
    "question": "What is the predicted operational strain over four hours?",
    "options": [
        {"id": "low", "description": "Capacity should comfortably exceed demand."},
        {"id": "moderate", "description": "Demand may temporarily approach capacity."},
        {"id": "high", "description": "Demand is likely to exceed staffed capacity."},
        {"id": "insufficient", "description": "The evidence is insufficient."},
    ],
}


@pytest.fixture(scope="module")
def engine():
    if not MODEL_DIR:
        pytest.skip("no edge0 checkpoint; set EDGE0_35B_MODEL")
    from edge0 import AutoEngine
    eng = AutoEngine.from_pretrained(MODEL_DIR, name="edge0-35b")
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def scorer(engine):
    from jevedge0.decision.scorer import Edge0DecisionScorer
    return Edge0DecisionScorer(engine)


# ---- decision readout ----------------------------------------------------

def test_decision_returns_valid_distribution(scorer):
    record = scorer.score(ROW)
    assert record["choice"] in [o["id"] for o in ROW["options"]]
    assert pytest.approx(sum(record["probabilities"]), abs=1e-6) == 1.0
    assert len(record["probabilities"]) == len(ROW["options"])
    assert 0.0 <= record["margin"] <= 1.0
    assert record["entropy"] >= 0.0
    assert record["method"] == "option_logit_readout"


def test_decision_generates_no_text(scorer):
    """The readout must be one prefill, not a generation."""
    record = scorer.score(ROW)
    # A prefill-only pass reports the prompt length and no completion.
    assert record["input_tokens"] > 0
    assert "completion_tokens" not in record


def test_decision_is_deterministic(scorer):
    """Identical input must give identical logits: no sampling involved."""
    first = scorer.score(ROW)
    second = scorer.score(ROW)
    assert first["choice"] == second["choice"]
    for a, b in zip(first["option_logits"], second["option_logits"]):
        assert a == pytest.approx(b, abs=1e-3)


def test_decision_responds_to_evidence(scorer):
    """Changing the evidence must change the distribution."""
    strained = scorer.score(ROW)
    calm = scorer.score({
        **ROW, "id": "calm",
        "state": ("Current ED census is 12 with no admitted patients "
                  "boarding, no waiting room queue, and nursing staffing "
                  "one position above plan."),
    })
    assert strained["choice"] != calm["choice"] or (
        strained["probabilities"][2] > calm["probabilities"][2])


def test_permutation_stability_maps_back_to_option_ids(scorer):
    from jevedge0.decision.stability import score_with_stability
    record = score_with_stability(scorer, ROW, trials=3)
    stability = record["stability"]
    assert stability["trials"] == 3
    assert set(stability["mean_probabilities"]) == {
        o["id"] for o in ROW["options"]}
    assert 0.0 <= stability["max_probability_spread"] <= 1.0
    assert len(stability["winners"]) == 3


def test_rejects_multi_token_option_letters(scorer):
    """A row with too many options must raise, never silently truncate."""
    from jevedge0.decision.prompt import DecisionError
    too_many = {**ROW, "options": [
        {"id": f"o{i}", "description": f"Option {i}."} for i in range(20)]}
    with pytest.raises(DecisionError):
        scorer.score(too_many)


# ---- embeddings ----------------------------------------------------------

def test_embedder_parity():
    from jevedge0.rag.embed import BertEmbedder
    report = BertEmbedder().verify_parity()
    assert report["dimension"] == 384
    assert report["unit_norm"]
    assert report["separates_meaning"]
    assert report["batch_stable"]


def test_embeddings_rank_related_text_higher():
    import numpy as np
    from jevedge0.rag.embed import get_embedder
    embedder = get_embedder()
    query = embedder.encode_one("emergency department overcrowding")
    corpus = embedder.encode([
        "The ED is over capacity with patients boarding in hallways.",
        "Quarterly parking revenue increased by four percent.",
    ])
    scores = corpus @ query
    assert scores[0] > scores[1]


# ---- comparator ----------------------------------------------------------

def test_comparator_emits_an_action(engine, scorer):
    from edge0.server.chat import QueueServer
    from jevedge0.comparator.dual import Comparator
    from jevedge0.orchestrator.engine_client import InProcessClient

    client = InProcessClient(QueueServer(engine, model_name="edge0-35b"))
    result = Comparator(scorer, client.chat, stability_trials=2).run(ROW)
    assert result["action"] in ("act", "ask", "abstain", "escalate")
    assert isinstance(result["reasons"], list)
    assert "probability_status" in result
    if result["action"] == "act":
        assert result["agreement"] and not result["reasons"]
