"""Typed-decision prompt construction and answer-slot validation.

The prompt contract is deliberately identical to the JEV-CPU / SemIf
baseline (``semif_phase1/core.py`` + ``direct.py``): same system line,
same JSON user payload, same ``LETTERS`` alphabet, same strict
single-token round-trip validation.  Keeping it byte-identical is what
makes the Edge0-35B readout comparable to the 0.6B / 4B baselines at
all — a reworded prompt would confound the model comparison with a
prompt comparison.

Nothing here truncates, repairs, or guesses.  A row that cannot be
rendered into exactly-one-token answer slots raises; the caller decides
what to do about it.
"""

from __future__ import annotations

import hashlib
import json
import math

# Baseline alphabet (semif_phase1.core.LETTERS).  Two to sixteen options.
LETTERS = "ABCDEFGHIJKLMNOP"

# Baseline system line, verbatim.
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. "
    "Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

PROMPT_VERSION = "direct-options-v1"


class DecisionError(ValueError):
    """Raised when a decision row or its tokenization is unusable."""


def validate_row(row: dict) -> None:
    """Validate one decision row (ported from semif_phase1.core).

    A row needs ``id``, ``state``, ``question`` and 2..16 ``options``,
    each option carrying a unique string ``id`` and a ``description``.
    """
    required = {"id", "state", "question", "options"}
    missing = required - row.keys()
    if missing:
        raise DecisionError(f"row is missing fields: {sorted(missing)}")
    for key in ("id", "question"):
        if not isinstance(row[key], str) or not row[key]:
            raise DecisionError(f"{key} must be a nonempty string")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise DecisionError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DecisionError(
            "state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise DecisionError(
            f"options must contain 2-{len(LETTERS)} entries, got "
            f"{len(options) if isinstance(options, list) else type(options)}")
    ids = []
    for option in options:
        if (not isinstance(option, dict)
                or not isinstance(option.get("id"), str)
                or not isinstance(option.get("description"), str)):
            raise DecisionError(
                "each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise DecisionError("option ids must be unique")


def direct_messages(row: dict) -> list[dict]:
    """Render the chat messages for one decision (baseline-identical)."""
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[i], "description": opt["description"]}
            for i, opt in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def render_prompt(tokenizer, row: dict) -> str:
    """Apply the checkpoint's chat template in no-thinking mode.

    Thinking must be off: the readout reads the logits at the position
    where the answer letter belongs, and a ``<think>`` opener would put
    reasoning tokens there instead.
    """
    messages = direct_messages(row)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        # Template without the kwarg.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def slot_ids(tokenizer, count: int) -> list[int]:
    """Token ids for the first ``count`` answer letters.

    Each letter must encode to exactly one token that decodes back to
    itself, and the set must not collide.  This is the property the whole
    readout rests on: if a letter were two tokens, the last-position
    logits would not be scoring that option at all.
    """
    if not 2 <= count <= len(LETTERS):
        raise DecisionError(f"need 2-{len(LETTERS)} options, got {count}")
    result: list[int] = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise DecisionError(
                f"answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise DecisionError("answer-slot tokens collide")
    return result


def encode_prompt(tokenizer, row: dict,
                  max_tokens: int = 8192) -> tuple[list[int], list[int], str]:
    """Encode one decision; return ``(prompt_ids, slot_ids, prompt_sha256)``.

    Also checks answer-boundary invariance: appending the answer letter
    must extend the tokenization by exactly that letter's token rather
    than re-tokenizing the join.  Without this check a prompt ending in
    certain characters can merge with the letter, and the slot logits
    would silently score the wrong thing.
    """
    prompt = render_prompt(tokenizer, row)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise DecisionError(f"row {row['id']}: empty prompt encoding")
    if len(ids) > max_tokens:
        raise DecisionError(
            f"row {row['id']}: {len(ids)} input tokens exceed limit "
            f"{max_tokens}; no truncation allowed")
    slots = slot_ids(tokenizer, len(row["options"]))
    for letter, token in zip(LETTERS, slots):
        joined = tokenizer.encode(prompt + letter, add_special_tokens=False)
        if joined != ids + [token]:
            raise DecisionError(
                f"row {row['id']}: answer boundary changes tokenization for "
                f"slot {letter}")
    return ids, slots, digest(prompt)


def softmax(values: list[float]) -> list[float]:
    """Softmax restricted to the declared option logits."""
    if len(values) < 2 or any(not math.isfinite(v) for v in values):
        raise DecisionError("need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(v - maximum) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def entropy(probabilities: list[float]) -> float:
    """Shannon entropy in nats over the option distribution."""
    return -sum(p * math.log(p) for p in probabilities if p > 0.0)


def margin(probabilities: list[float]) -> float:
    """Top-two probability margin (decisiveness, not correctness)."""
    if len(probabilities) < 2:
        raise DecisionError("need at least two probabilities")
    top = sorted(probabilities, reverse=True)
    return top[0] - top[1]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
