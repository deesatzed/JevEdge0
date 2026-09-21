"""Noul and ordinal Score: the two typed-decision primitives besides Choice.

The project's decision core (``decision/scorer.py``) implements one
primitive -- an unordered multiple-choice readout -- and calls it
``Choice`` implicitly by never distinguishing it from anything else.
Two other primitives compose naturally from the same option-logit
readout, changing only how the resulting distribution is aggregated:

``score_noul``     a yes/no question with a probability, not just a bool.
``score_ordinal``  an ORDERED rubric (e.g. severity levels), returning a
                   probability-weighted position rather than a bucket --
                   1.8 on a 0..2 scale genuinely means "between level 1
                   and level 2", which a bare argmax cannot express.

Both are built on ``Edge0DecisionScorer.score()`` and inherit its
properties unchanged: one prefill, no text generated, and a result that
is a conditional option score, never calibrated confidence.
"""

from __future__ import annotations

from jevedge0.decision.prompt import DecisionError

DEFAULT_YES = "Yes."
DEFAULT_NO = "No."


def score_noul(scorer, state, question: str, yes: str = DEFAULT_YES,
               no: str = DEFAULT_NO, row_id: str = "noul") -> dict:
    """Score a yes/no question. Returns a bool plus its probability.

    ``scorer`` is anything with a ``score(row) -> dict`` method matching
    ``Edge0DecisionScorer`` (a fake works fine, as in the guard tests).
    ``state`` is the evidence; ``question`` the criterion. ``yes``/``no``
    let the two options be phrased for the specific question rather than
    forced into generic wording when that would read oddly.

    Returns ``value`` (bool, True iff yes was more probable), ``probability``
    (of the *true* branch specifically, not of whichever won), ``margin``,
    ``entropy``, and ``legal_mass`` passed through from the underlying
    decision record.
    """
    row = {
        "id": row_id,
        "state": state,
        "question": question,
        "options": [
            {"id": "yes", "description": yes},
            {"id": "no", "description": no},
        ],
    }
    record = scorer.score(row)
    probabilities = dict(zip(record["option_ids"], record["probabilities"]))
    return {
        "value": record["choice"] == "yes",
        "probability": probabilities["yes"],
        "margin": record["margin"],
        "entropy": record["entropy"],
        "legal_mass": record.get("legal_mass"),
        "probabilities": probabilities,
        "record": record,
    }


def score_ordinal(scorer, state, question: str, levels: list[dict],
                  row_id: str = "ordinal") -> dict:
    """Score an ORDERED rubric, returning a probability-weighted position.

    ``levels`` must be given in ascending order, e.g.::

        [{"id": "calm", "description": "..."},
         {"id": "concerned", "description": "..."},
         {"id": "angry", "description": "..."}]

    The distinguishing feature versus an unordered Choice over the same
    options is ``expected_score``: rather than reporting only the single
    most probable level, this computes ``sum(i * p_i)`` over the level
    indices. A distribution split between "concerned" (index 1) and
    "angry" (index 2) yields something like 1.8 -- genuinely between the
    two levels -- which is information an argmax alone discards entirely.
    This only means anything because the levels are ordered; the same
    computation over an unordered Choice's arbitrary option indices would
    be meaningless arithmetic on category labels.

    Returns ``expected_score`` (real-valued position on the level index
    scale), ``expected_normalized`` (rescaled to [0, 1] by level count),
    ``argmax_level`` (id of the single most probable level, for callers
    that still want a discrete answer), ``distribution`` (id ->
    probability), and ``legal_mass``.
    """
    if len(levels) < 2:
        raise DecisionError(
            f"score_ordinal needs at least 2 ordered levels, got {len(levels)}")

    row = {"id": row_id, "state": state, "question": question,
          "options": levels}
    record = scorer.score(row)

    level_ids = [level["id"] for level in levels]
    distribution = dict(zip(record["option_ids"], record["probabilities"]))
    # record["option_ids"] preserves the declared order (decision/prompt.py
    # assigns letters in the order options were given), so indexing by
    # position in `levels` is safe -- but look up by id explicitly rather
    # than assuming order, since a scorer implementation is free to
    # reorder option_ids and this must not silently miscompute.
    ordered_probabilities = [distribution[level_id] for level_id in level_ids]

    expected_score = sum(i * p for i, p in enumerate(ordered_probabilities))
    denominator = len(levels) - 1
    expected_normalized = expected_score / denominator if denominator else 0.0
    argmax_index = max(range(len(level_ids)),
                       key=lambda i: ordered_probabilities[i])

    return {
        "expected_score": expected_score,
        "expected_normalized": expected_normalized,
        "argmax_level": level_ids[argmax_index],
        "distribution": distribution,
        "margin": record["margin"],
        "entropy": record["entropy"],
        "legal_mass": record.get("legal_mass"),
        "record": record,
    }
