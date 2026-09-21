"""Split-conformal prediction for typed decisions.

Replaces a hand-set confidence threshold with a statistically valid one.
Given a labeled calibration set, this computes a threshold such that the
prediction *set* it produces contains the true label with probability at
least ``1 - alpha``, over the distribution the calibration set was drawn
from -- regardless of how badly calibrated the underlying softmax scores
are. This is the point of conformal prediction: it does not require the
model's probabilities to mean anything in an absolute sense, only that
they rank options consistently.

The nonconformity score used is the standard LAC ("least ambiguous set-
valued classifier") score: ``s = 1 - p[true_label]``. A low score means
the model put high probability on the correct answer; a high score means
it did not. The calibration step finds the empirical quantile of these
scores across the labeled set, and every future prediction includes every
option whose score does not exceed that quantile -- i.e. every option
the model was at least as confident about as the calibration data implies
it should be for the guarantee to hold.

IMPORTANT ON WHAT THE GUARANTEE IS: coverage holds only for inputs drawn
from the same distribution as the calibration set. It is a guarantee
about the *procedure*, applied repeatedly over that distribution -- not a
guarantee that any individual answer is correct. A single confident-looking
wrong answer inside a valid 90% coverage guarantee is not a contradiction;
it is one of the roughly 10% the guarantee always allows for.
"""

from __future__ import annotations

import json
import math
import time

MIN_CALIBRATION_ROWS = 20


class ConformalError(ValueError):
    """Raised when calibration or prediction cannot proceed safely."""


def _nonconformity_scores(records: list[dict], labels: list[str]) -> list[float]:
    """Compute the LAC score ``1 - p[true_label]`` for each record.

    ``records`` are decision-scorer outputs (each with ``option_ids`` and
    ``probabilities``, in matching order); ``labels`` is the true option id
    for each, in the same order as ``records``.
    """
    scores = []
    for record, label in zip(records, labels):
        option_ids = record["option_ids"]
        probabilities = record["probabilities"]
        if label not in option_ids:
            raise ConformalError(
                f"label {label!r} is not among this record's option_ids "
                f"{option_ids!r}")
        p_true = probabilities[option_ids.index(label)]
        scores.append(1.0 - p_true)
    return scores


def calibrate(records: list[dict], labels: list[str],
             alpha: float = 0.1) -> dict:
    """Compute a conformal threshold from labeled calibration records.

    ``records`` must be the *raw scorer output* (``Edge0DecisionScorer
    .score()`` records or equivalent), not just probability lists, so the
    option ordering is unambiguous. ``labels`` is the true option id for
    each record, in the same order.

    Returns a calibration dict: ``n``, ``alpha``, ``q_hat`` (the
    threshold), and the scores used, so the artifact is fully
    reconstructable and auditable.
    """
    if len(records) != len(labels):
        raise ConformalError(
            f"records ({len(records)}) and labels ({len(labels)}) must be "
            "the same length")
    n = len(records)
    if n < MIN_CALIBRATION_ROWS:
        raise ConformalError(
            f"calibrating on {n} rows is not enough to produce a "
            f"meaningful quantile; need at least {MIN_CALIBRATION_ROWS}")
    if not 0.0 < alpha < 1.0:
        raise ConformalError(f"alpha must be in (0, 1), got {alpha}")

    scores = sorted(_nonconformity_scores(records, labels))
    # Standard split-conformal quantile: ceil((n+1)(1-alpha)) / n, clipped
    # to the largest available score when the rank would exceed n (which
    # happens for small n / large 1-alpha -- the guarantee then requires
    # the full range of observed nonconformity, i.e. q_hat = 1.0).
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank >= n:
        q_hat = 1.0
    else:
        q_hat = scores[rank - 1]

    return {
        "n": n,
        "alpha": alpha,
        "q_hat": q_hat,
        "scores": scores,
    }


def predict_set(probabilities: list[float], option_ids: list[str],
                calibration: dict) -> dict:
    """Compute the conformal prediction set for one decision.

    ``probabilities`` and ``option_ids`` come from a decision record (same
    order). ``calibration`` is the dict returned by ``calibrate``.

    Returns ``set`` (the option ids whose probability clears the
    threshold), ``set_size``, ``singleton`` (True iff exactly one option
    is included), ``abstain`` (True iff the set is not a singleton --
    either ambiguous, with more than one plausible option, or, in a
    degenerate case, empty), and the ``alpha`` used.
    """
    q_hat = calibration["q_hat"]
    alpha = calibration["alpha"]
    threshold = 1.0 - q_hat
    included = [oid for oid, p in zip(option_ids, probabilities)
               if p >= threshold]
    return {
        "set": included,
        "set_size": len(included),
        "singleton": len(included) == 1,
        "abstain": len(included) != 1,
        "alpha": alpha,
        "threshold": threshold,
    }


def empirical_coverage(records: list[dict], labels: list[str],
                       calibration: dict) -> dict:
    """Measure realized coverage of a calibration on held-out data.

    Coverage is the fraction of records whose prediction set actually
    contains the true label. This must be measured on data disjoint from
    what produced ``calibration`` -- using the same rows for both
    calibrating and measuring coverage would be circular and would not
    test anything.

    Returns the measured coverage alongside ``n`` and ``mean_set_size``,
    so both numbers this stage is required to report are available from
    one call.
    """
    if not records:
        raise ConformalError("no held-out records to measure coverage on")
    covered = 0
    set_sizes = []
    for record, label in zip(records, labels):
        result = predict_set(record["probabilities"], record["option_ids"],
                            calibration)
        set_sizes.append(result["set_size"])
        if label in result["set"]:
            covered += 1
    return {
        "n": len(records),
        "covered": covered,
        "coverage": covered / len(records),
        "mean_set_size": sum(set_sizes) / len(set_sizes),
        "target": 1 - calibration["alpha"],
    }


def save_calibration(calibration: dict, path: str,
                     coverage_check: dict | None = None) -> None:
    """Persist a calibration with the honesty caveats the spec requires.

    Always written: ``n``, ``alpha``, ``q_hat``, the date, and the exact
    coverage caveat. ``coverage_check`` (from ``empirical_coverage``), when
    supplied, is included as-is -- including an unfavorable result. This
    function does not filter or adjust what it is given.
    """
    artifact = {
        "n": calibration["n"],
        "alpha": calibration["alpha"],
        "q_hat": calibration["q_hat"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caveat": (
            "Coverage holds only for inputs drawn from the same "
            "distribution as the calibration set. It is a guarantee about "
            "the procedure, not a guarantee that any individual answer is "
            "correct."),
    }
    if coverage_check is not None:
        artifact["empirical_coverage_check"] = coverage_check
    with open(path, "w") as fh:
        json.dump(artifact, fh, indent=2)


def load_calibration(path: str) -> dict:
    """Load a calibration artifact previously written by save_calibration."""
    with open(path) as fh:
        data = json.load(fh)
    for key in ("n", "alpha", "q_hat"):
        if key not in data:
            raise ConformalError(
                f"calibration file {path!r} is missing required key {key!r}")
    return data
