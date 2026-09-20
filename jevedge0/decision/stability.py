"""Permutation and paraphrase stability for typed decisions.

A single forward pass gives one distribution over option letters.  That
distribution is sensitive to which letter an option happened to receive:
models carry position and letter biases, so "A" is not a neutral label.
A decision that flips when the options are reordered is an artifact of
presentation, not a judgment about the evidence.

This module re-scores a row under several option permutations, maps every
result back to stable option IDs, and reports whether the winner survived.
Callers should treat instability as a reason to abstain or escalate rather
than as noise to average away.
"""

from __future__ import annotations

import itertools
import random
import statistics

from jevedge0.decision.prompt import DecisionError, entropy, margin


def permute_row(row: dict, order: list[int]) -> dict:
    """Return ``row`` with its options reordered by ``order``."""
    options = row["options"]
    if sorted(order) != list(range(len(options))):
        raise DecisionError("permutation must be over all option indices")
    permuted = dict(row)
    permuted["options"] = [options[i] for i in order]
    return permuted


def permutations_for(n: int, trials: int,
                     seed: int | None = 0) -> list[list[int]]:
    """Choose option orderings to test.

    The identity ordering always comes first so the primary result is the
    canonical one.  For small option counts every ordering is enumerated
    (exhaustive beats sampling when it is cheap); larger counts sample
    distinct random orderings under a fixed seed for reproducibility.
    """
    identity = list(range(n))
    if trials <= 1:
        return [identity]
    total = 1
    for k in range(2, n + 1):
        total *= k
    if total <= trials:
        orders = [list(p) for p in itertools.permutations(identity)]
        orders.sort(key=lambda o: o != identity)
        return orders
    rng = random.Random(seed)
    seen = {tuple(identity)}
    orders = [identity]
    while len(orders) < trials:
        candidate = identity[:]
        rng.shuffle(candidate)
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        orders.append(candidate)
    return orders


def score_with_stability(scorer, row: dict, trials: int = 4,
                         seed: int | None = 0) -> dict:
    """Score ``row`` under several option orderings.

    Returns the canonical (identity-order) record extended with a
    ``stability`` block. Every per-trial distribution is remapped to the
    row's declared option IDs before aggregation, so the numbers are
    comparable across orderings.
    """
    options = row["options"]
    option_ids = [opt["id"] for opt in options]
    orders = permutations_for(len(options), trials, seed=seed)

    per_option: dict[str, list[float]] = {oid: [] for oid in option_ids}
    winners: list[str] = []
    trial_records = []

    canonical = None
    for order in orders:
        record = scorer.score(permute_row(row, order))
        # record["option_ids"] is in permuted order; map back.
        remapped = dict(zip(record["option_ids"], record["probabilities"]))
        for oid in option_ids:
            per_option[oid].append(remapped[oid])
        winners.append(record["choice"])
        trial_records.append({
            "order": order,
            "choice": record["choice"],
            "probabilities": remapped,
        })
        if order == list(range(len(options))):
            canonical = record

    if canonical is None:  # pragma: no cover - identity is always included
        raise DecisionError("canonical ordering was not scored")

    mean_probs = [statistics.fmean(per_option[oid]) for oid in option_ids]
    spreads = {oid: (max(v) - min(v)) for oid, v in per_option.items()}
    unique_winners = sorted(set(winners))
    majority = max(unique_winners, key=winners.count)

    result = dict(canonical)
    result["stability"] = {
        "trials": len(orders),
        "stable_across_permutations": len(unique_winners) == 1,
        "winners": winners,
        "majority_choice": majority,
        "majority_fraction": winners.count(majority) / len(winners),
        "mean_probabilities": dict(zip(option_ids, mean_probs)),
        "max_probability_spread": max(spreads.values()),
        "probability_spread": spreads,
        "mean_margin": margin(mean_probs),
        "mean_entropy": entropy(mean_probs),
        "trial_records": trial_records,
    }
    return result


def paraphrase_rows(row: dict, criteria: list[str]) -> list[dict]:
    """Build rows that re-ask the same decision with reworded criteria.

    The paraphrases are supplied by the caller: this module will not
    invent rewordings, because an auto-generated paraphrase that changes
    the meaning would silently turn a stability test into a different
    question.
    """
    rows = []
    for index, text in enumerate(criteria):
        if not isinstance(text, str) or not text:
            raise DecisionError("each paraphrase must be a nonempty string")
        variant = dict(row)
        variant["id"] = f"{row['id']}::paraphrase-{index}"
        variant["question"] = text
        rows.append(variant)
    return rows


def paraphrase_stability(scorer, row: dict, criteria: list[str]) -> dict:
    """Score the row under caller-supplied paraphrased criteria."""
    rows = paraphrase_rows(row, criteria)
    records = [scorer.score(r) for r in rows]
    winners = [r["choice"] for r in records]
    unique = sorted(set(winners))
    return {
        "variants": len(records),
        "stable_across_paraphrases": len(unique) == 1,
        "winners": winners,
        "records": records,
    }
