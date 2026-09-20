"""Two-path comparator: generative judgment against typed decision.

The design from ``jevadd.md``: run the same problem through Edge0 twice,
once as open generation and once as a constrained option-logit readout,
then compare.  The channels fail differently — generation is fluent and
can rationalize a wrong answer at length, while the logit readout is
bounded but blind to nuance it was never asked about.  Agreement between
two channels that fail differently is worth more than either alone, and
*disagreement* is the genuinely useful signal: it marks the cases where a
human should look.

The guardrail deliberately refuses to convert this into a single
confidence number.  It emits an action — act, ask, abstain, escalate —
because that is the decision the operator actually faces.
"""

from __future__ import annotations

import json
import re

from jevedge0.decision.stability import score_with_stability

ACT = "act"
ASK = "ask"
ABSTAIN = "abstain"
ESCALATE = "escalate"

JUDGMENT_SYSTEM = (
    "Apply the criterion to the evidence and use your best judgment.\n"
    "\n"
    "First reason about what the evidence does and does not establish, "
    "including anything important that is missing. Then state which single "
    "option you would choose.\n"
    "\n"
    "End your reply with exactly one final line:\n"
    "CHOICE: <option_id>\n"
    "\n"
    "The option_id must be copied exactly from the listed options. If the "
    "evidence cannot support a choice, use the option that expresses "
    "insufficiency when one is offered."
)


def judgment_messages(row: dict) -> list[dict]:
    """Messages for the generative channel (ids, not letters).

    The generative channel is shown option *ids* while the logit channel
    is shown *letters*.  That asymmetry is deliberate: it keeps the two
    channels from sharing a presentation artifact, so agreement is about
    the evidence rather than about the same letter bias appearing twice.
    """
    options = "\n".join(
        f"- {opt['id']}: {opt['description']}" for opt in row["options"])
    state = row["state"]
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": JUDGMENT_SYSTEM},
        {"role": "user", "content": (
            f"Evidence:\n{state}\n\n"
            f"Criterion:\n{row['question']}\n\n"
            f"Options:\n{options}")},
    ]


def parse_judgment(text: str, option_ids: list[str]) -> dict:
    """Extract the chosen option id from a generative reply."""
    reasoning = text.strip()
    match = None
    for candidate in re.finditer(r"CHOICE:\s*([^\n]+)", text, re.IGNORECASE):
        match = candidate
    if match:
        raw = match.group(1).strip().strip("`\"'.,")
        for option_id in option_ids:
            if raw.lower() == option_id.lower():
                return {"choice": option_id, "reasoning": reasoning,
                        "parsed": True}
        for option_id in option_ids:
            if option_id.lower() in raw.lower():
                return {"choice": option_id, "reasoning": reasoning,
                        "parsed": True}
    # No usable marker: look for a unique option mention in the tail.
    tail = text[-400:].lower()
    mentioned = [o for o in option_ids if o.lower() in tail]
    if len(mentioned) == 1:
        return {"choice": mentioned[0], "reasoning": reasoning,
                "parsed": True}
    return {"choice": None, "reasoning": reasoning, "parsed": False}


class Comparator:
    """Run both channels and apply the guardrail."""

    def __init__(self, scorer, chat_fn, stability_trials: int = 4,
                 min_margin: float = 0.20, max_entropy: float = 1.00,
                 max_spread: float = 0.25):
        self.scorer = scorer
        self.chat_fn = chat_fn
        self.stability_trials = stability_trials
        self.min_margin = min_margin
        self.max_entropy = max_entropy
        self.max_spread = max_spread

    def run(self, row: dict) -> dict:
        option_ids = [o["id"] for o in row["options"]]

        decision = score_with_stability(
            self.scorer, row, trials=self.stability_trials)
        judgment_text = self.chat_fn(judgment_messages(row))
        judgment = parse_judgment(judgment_text, option_ids)

        agree = (judgment["choice"] is not None
                 and judgment["choice"] == decision["choice"])
        verdict = self.guardrail(decision, judgment, agree)
        return {
            "row_id": row["id"],
            "decision": decision,
            "judgment": judgment,
            "agreement": agree,
            **verdict,
        }

    def guardrail(self, decision: dict, judgment: dict, agree: bool) -> dict:
        """Choose act / ask / abstain / escalate and say why."""
        stability = decision.get("stability", {})
        stable = stability.get("stable_across_permutations", True)
        spread = stability.get("max_probability_spread", 0.0)
        margin = decision["margin"]
        entropy = decision["entropy"]
        reasons = []

        if not judgment["parsed"]:
            reasons.append("the generative channel did not state a usable "
                           "choice")
        if not agree and judgment["parsed"]:
            reasons.append(
                f"channels disagree: decision={decision['choice']!r}, "
                f"judgment={judgment['choice']!r}")
        if not stable:
            reasons.append(
                "the choice changes when the options are reordered "
                f"(winners: {', '.join(sorted(set(stability.get('winners', []))))})")
        if spread > self.max_spread:
            reasons.append(f"probabilities move {spread:.2f} across "
                           f"orderings (limit {self.max_spread:.2f})")
        if margin < self.min_margin:
            reasons.append(f"top-two margin {margin:.2f} is below "
                           f"{self.min_margin:.2f}")
        if entropy > self.max_entropy:
            reasons.append(f"distribution entropy {entropy:.2f} exceeds "
                           f"{self.max_entropy:.2f}")

        if not reasons:
            action = ACT
            summary = (f"Both channels chose {decision['choice']!r}; the "
                       "choice held across reorderings.")
        elif not stable and not agree:
            action = ESCALATE
            summary = ("Unstable and contested: neither channel gives a "
                       "dependable answer.")
        elif not agree and judgment["parsed"]:
            action = ESCALATE
            summary = "The two channels reached different conclusions."
        elif margin < self.min_margin or entropy > self.max_entropy:
            action = ABSTAIN
            summary = "The distribution is too flat to call."
        else:
            action = ASK
            summary = "The result needs confirmation before it is acted on."

        return {
            "action": action,
            "summary": summary,
            "reasons": reasons,
            "checks": {
                "agreement": agree,
                "judgment_parsed": judgment["parsed"],
                "stable_across_permutations": stable,
                "max_probability_spread": spread,
                "margin": margin,
                "entropy": entropy,
            },
            "thresholds": {
                "min_margin": self.min_margin,
                "max_entropy": self.max_entropy,
                "max_spread": self.max_spread,
            },
            "probability_status": decision["probability_status"],
        }
