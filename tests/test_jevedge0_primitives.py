"""Tests for the Noul and ordinal Score typed-decision primitives.

No model weights needed: a fake scorer plays Edge0DecisionScorer's role,
exactly as in test_jevedge0_guard.py. legal_mass on real weights is
validated separately (GOAL_ENHANCE.md Sec 7.C).
"""

from __future__ import annotations

import pytest

from jevedge0.decision.prompt import DecisionError
from jevedge0.decision.primitives import score_noul, score_ordinal


class FakeScorer:
    """Scripted scorer returning a fixed distribution over given options."""

    def __init__(self, probabilities: dict[str, float], legal_mass=1.0):
        self.probabilities = probabilities
        self.legal_mass = legal_mass

    def score(self, row):
        option_ids = [o["id"] for o in row["options"]]
        probs = [self.probabilities[oid] for oid in option_ids]
        best = option_ids[max(range(len(probs)), key=probs.__getitem__)]
        sorted_probs = sorted(probs, reverse=True)
        margin = (sorted_probs[0] - sorted_probs[1]
                  if len(sorted_probs) > 1 else 1.0)
        return {
            "id": row["id"], "choice": best, "option_ids": option_ids,
            "probabilities": probs, "margin": margin, "entropy": 0.0,
            "legal_mass": self.legal_mass,
        }


# ---- 1. Noul returns True/False by which option dominates -----------------

def test_noul_true_when_yes_dominates():
    scorer = FakeScorer({"yes": 0.9, "no": 0.1})
    result = score_noul(scorer, "state", "is this safe?")
    assert result["value"] is True
    assert result["probability"] == pytest.approx(0.9)


def test_noul_false_when_no_dominates():
    scorer = FakeScorer({"yes": 0.2, "no": 0.8})
    result = score_noul(scorer, "state", "is this safe?")
    assert result["value"] is False
    assert result["probability"] == pytest.approx(0.2)  # probability of YES specifically


# ---- 2. Noul probabilities sum to 1 ----------------------------------------

def test_noul_probabilities_sum_to_one():
    scorer = FakeScorer({"yes": 0.37, "no": 0.63})
    result = score_noul(scorer, "state", "q")
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_noul_custom_wording():
    scorer = FakeScorer({"yes": 0.6, "no": 0.4})
    result = score_noul(scorer, "state", "escalate?",
                        yes="Escalate immediately.", no="Continue monitoring.")
    assert result["value"] is True


def test_noul_carries_legal_mass_through():
    scorer = FakeScorer({"yes": 0.5, "no": 0.5}, legal_mass=0.73)
    result = score_noul(scorer, "state", "q")
    assert result["legal_mass"] == pytest.approx(0.73)


# ---- 3. ordinal expected score matches a hand-computed value --------------

def test_ordinal_expected_score_matches_hand_computation():
    levels = [{"id": "calm", "description": "d"},
             {"id": "concerned", "description": "d"},
             {"id": "angry", "description": "d"}]
    # indices: calm=0, concerned=1, angry=2
    # expected = 0*0.1 + 1*0.3 + 2*0.6 = 0 + 0.3 + 1.2 = 1.5
    scorer = FakeScorer({"calm": 0.1, "concerned": 0.3, "angry": 0.6})
    result = score_ordinal(scorer, "state", "how upset?", levels)
    assert result["expected_score"] == pytest.approx(1.5)


def test_ordinal_expected_score_between_concerned_and_angry_example():
    """The example from the spec: distribution split between the top two
    levels should read as 'between concerned and angry', not as either
    discretely."""
    levels = [{"id": "calm", "description": "d"},
             {"id": "concerned", "description": "d"},
             {"id": "angry", "description": "d"}]
    # concerned=0.2, angry=0.8, calm=0.0 -> expected = 1*0.2 + 2*0.8 = 1.8
    scorer = FakeScorer({"calm": 0.0, "concerned": 0.2, "angry": 0.8})
    result = score_ordinal(scorer, "state", "how upset?", levels)
    assert result["expected_score"] == pytest.approx(1.8)
    assert 1.0 < result["expected_score"] < 2.0


# ---- 4. concentrated distributions -> normalized near 1.0 / 0.0 -----------

def test_ordinal_concentrated_on_top_level_normalizes_near_one():
    levels = [{"id": "low", "description": "d"}, {"id": "mid", "description": "d"},
             {"id": "high", "description": "d"}]
    scorer = FakeScorer({"low": 0.001, "mid": 0.001, "high": 0.998})
    result = score_ordinal(scorer, "state", "q", levels)
    assert result["expected_normalized"] == pytest.approx(1.0, abs=0.01)


def test_ordinal_concentrated_on_bottom_level_normalizes_near_zero():
    levels = [{"id": "low", "description": "d"}, {"id": "mid", "description": "d"},
             {"id": "high", "description": "d"}]
    scorer = FakeScorer({"low": 0.998, "mid": 0.001, "high": 0.001})
    result = score_ordinal(scorer, "state", "q", levels)
    assert result["expected_normalized"] == pytest.approx(0.0, abs=0.01)


# ---- 5. expected_score strictly between two levels with split probability --

def test_ordinal_expected_score_strictly_between_split_levels():
    levels = [{"id": "a", "description": "d"}, {"id": "b", "description": "d"}]
    scorer = FakeScorer({"a": 0.5, "b": 0.5})
    result = score_ordinal(scorer, "state", "q", levels)
    assert 0.0 < result["expected_score"] < 1.0
    assert result["expected_score"] == pytest.approx(0.5)


@pytest.mark.parametrize("p_a,p_b", [(0.9, 0.1), (0.1, 0.9), (0.3, 0.7)])
def test_ordinal_expected_score_always_between_levels_for_two_level_split(p_a, p_b):
    levels = [{"id": "a", "description": "d"}, {"id": "b", "description": "d"}]
    scorer = FakeScorer({"a": p_a, "b": p_b})
    result = score_ordinal(scorer, "state", "q", levels)
    assert 0.0 <= result["expected_score"] <= 1.0


# ---- 6. legal_mass properties ----------------------------------------------

def test_ordinal_legal_mass_bounded_and_carried_through():
    levels = [{"id": "a", "description": "d"}, {"id": "b", "description": "d"}]
    scorer = FakeScorer({"a": 0.5, "b": 0.5}, legal_mass=0.42)
    result = score_ordinal(scorer, "state", "q", levels)
    assert 0.0 <= result["legal_mass"] <= 1.0
    assert result["legal_mass"] == pytest.approx(0.42)


def test_legal_mass_higher_for_well_posed_than_badly_posed_question():
    """Not a claim about the scorer's internal computation (that needs
    real weights, tested in Sec 7.C) -- this only checks that the
    primitives correctly surface whatever legal_mass the scorer reports,
    so a caller comparing two real runs sees the real difference."""
    levels = [{"id": "a", "description": "d"}, {"id": "b", "description": "d"}]
    well_posed = FakeScorer({"a": 0.5, "b": 0.5}, legal_mass=0.95)
    badly_posed = FakeScorer({"a": 0.5, "b": 0.5}, legal_mass=0.15)
    result_well = score_ordinal(well_posed, "state", "q", levels)
    result_bad = score_ordinal(badly_posed, "state", "q", levels)
    assert result_well["legal_mass"] > result_bad["legal_mass"]


# ---- 7. ordinal with fewer than two levels raises --------------------------

def test_ordinal_with_zero_levels_raises():
    scorer = FakeScorer({})
    with pytest.raises(DecisionError, match="at least 2"):
        score_ordinal(scorer, "state", "q", [])


def test_ordinal_with_one_level_raises():
    scorer = FakeScorer({"only": 1.0})
    with pytest.raises(DecisionError, match="at least 2"):
        score_ordinal(scorer, "state", "q",
                      [{"id": "only", "description": "d"}])


# ---- distribution ordering integrity ---------------------------------------

def test_ordinal_indexes_by_id_not_by_returned_order():
    """If a scorer implementation ever returns option_ids in a different
    order than declared, expected_score must still be computed against
    the CALLER's declared level order, not whatever order came back."""
    levels = [{"id": "low", "description": "d"}, {"id": "high", "description": "d"}]

    class ReorderingScorer:
        def score(self, row):
            # Deliberately reverse the order in the returned record.
            option_ids = [o["id"] for o in row["options"]][::-1]
            probs = [0.9, 0.1]  # high=0.9, low=0.1 in this reversed order
            return {"id": row["id"], "choice": option_ids[0],
                    "option_ids": option_ids, "probabilities": probs,
                    "margin": 0.8, "entropy": 0.1, "legal_mass": 1.0}

    result = score_ordinal(ReorderingScorer(), "state", "q", levels)
    # low=0 index gets 0.1 (from the reversed mapping), high=1 index gets 0.9
    # expected = 0*0.1 + 1*0.9 = 0.9
    assert result["expected_score"] == pytest.approx(0.9)
    assert result["distribution"]["low"] == pytest.approx(0.1)
    assert result["distribution"]["high"] == pytest.approx(0.9)
