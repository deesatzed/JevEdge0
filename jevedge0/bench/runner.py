"""Benchmark harness comparing decision methods on identical rows.

The methods compared here all run locally on Edge0.  The JEV-CPU 0.6B and
MLX 4B baselines are produced by *that* project's own CLI — this harness
reads their results file and merges it rather than reimplementing their
scoring, so the baseline numbers are theirs, measured on their code.

What it reports, and does not report:

* Discrimination (accuracy against supplied labels) only when labels are
  supplied.  With no labels there is no accuracy, and the report says so
  rather than inventing a number.
* Calibration as binned observed-vs-stated frequency, presented as a
  diagnostic of the *readout*, never as a claim that the softmax values
  are probabilities of being right.
* Option-order sensitivity and abstention rate, which need no labels.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time


def _portable_path(path: str) -> str:
    """Render a path for a report that may be shared or committed.

    Relative to the working directory when it sits inside it, otherwise
    ``~``-prefixed. Either way the runner's home directory does not end
    up embedded in the artifact.
    """
    absolute = os.path.abspath(os.path.expanduser(path))
    try:
        relative = os.path.relpath(absolute, os.getcwd())
        if not relative.startswith(".."):
            return relative
    except ValueError:  # different drive on Windows
        pass
    home = os.path.expanduser("~")
    if absolute.startswith(home + os.sep):
        return "~" + absolute[len(home):]
    return absolute


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(os.path.expanduser(path)) as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: invalid JSON: {exc}")
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def load_labels(path: str | None) -> dict:
    if not path:
        return {}
    labels = {}
    with open(os.path.expanduser(path)) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            labels[item["id"]] = item["label"]
    return labels


def calibration_bins(pairs: list[tuple[float, bool]], bins: int = 5) -> list[dict]:
    """Bin (stated score, was correct) pairs.

    A well-behaved readout would put roughly ``mean_score`` fraction of
    each bin in the correct column.  Deviation is the evidence that the
    softmax is not a calibrated probability — which is the expected
    finding, and the reason to measure it.
    """
    if not pairs:
        return []
    out = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        bucket = [p for p in pairs
                  if low <= p[0] < high or (index == bins - 1 and p[0] == 1.0)]
        if not bucket:
            continue
        out.append({
            "range": f"{low:.1f}-{high:.1f}",
            "count": len(bucket),
            "mean_score": statistics.fmean(s for s, _ in bucket),
            "observed_accuracy": statistics.fmean(
                1.0 if c else 0.0 for _, c in bucket),
        })
    return out


def summarize(name: str, records: list[dict], labels: dict) -> dict:
    """Aggregate one method's records."""
    scored = [r for r in records if r.get("choice") is not None]
    unparsed = len(records) - len(scored)
    summary = {
        "method": name,
        "rows": len(records),
        "unparsed": unparsed,
        "abstained": sum(1 for r in scored
                         if str(r["choice"]).lower() in
                         ("insufficient", "cannot_determine", "unknown")),
    }

    timings = [r["forward_seconds"] for r in records if "forward_seconds" in r]
    if timings:
        summary["median_forward_seconds"] = statistics.median(timings)

    stabilities = [r["stability"]["stable_across_permutations"]
                   for r in records if "stability" in r]
    if stabilities:
        summary["permutation_stable_fraction"] = statistics.fmean(
            1.0 if s else 0.0 for s in stabilities)
        spreads = [r["stability"]["max_probability_spread"]
                   for r in records if "stability" in r]
        summary["median_probability_spread"] = statistics.median(spreads)

    if labels:
        matched = [r for r in scored if r["id"] in labels]
        if matched:
            correct = [r["choice"] == labels[r["id"]] for r in matched]
            summary["labeled_rows"] = len(matched)
            summary["accuracy"] = statistics.fmean(
                1.0 if c else 0.0 for c in correct)
            pairs = []
            for record, is_correct in zip(matched, correct):
                probabilities = record.get("probabilities")
                if isinstance(probabilities, dict):
                    top = max(probabilities.values())
                elif isinstance(probabilities, list) and probabilities:
                    top = max(probabilities)
                else:
                    continue
                pairs.append((top, is_correct))
            if pairs:
                summary["calibration"] = calibration_bins(pairs)
                summary["mean_top_score"] = statistics.fmean(
                    s for s, _ in pairs)
                summary["calibration_gap"] = (
                    summary["mean_top_score"] - summary["accuracy"])
    else:
        summary["accuracy"] = None
        summary["accuracy_note"] = (
            "no labels supplied; discrimination not measured")
    return summary


def run_benchmark(args) -> int:
    output = os.path.expanduser(args.output)
    if os.path.exists(output):
        raise SystemExit(
            f"{output} exists; benchmark output is create-only — pass a new "
            "filename")

    rows = load_rows(args.input)
    labels = load_labels(args.labels)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    known = {"edge0-logit", "edge0-judgment", "comparator"}
    unknown = set(methods) - known
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}; "
                         f"available: {sorted(known)}")

    from jevedge0.cli import _queue_server
    from jevedge0.comparator.dual import (Comparator, judgment_messages,
                                          parse_judgment)
    from jevedge0.decision.scorer import Edge0DecisionScorer
    from jevedge0.decision.stability import score_with_stability
    from jevedge0.orchestrator.engine_client import InProcessClient

    server = _queue_server(args)
    started = time.time()
    results: dict[str, list[dict]] = {}
    try:
        scorer = Edge0DecisionScorer(server.engine)
        client = InProcessClient(server)

        if "edge0-logit" in methods:
            records = []
            for index, row in enumerate(rows, start=1):
                print(f"[edge0-logit] {index}/{len(rows)} {row['id']}",
                      file=sys.stderr)
                record = (score_with_stability(scorer, row, trials=args.trials)
                          if args.trials > 1 else scorer.score(row))
                record["probabilities"] = dict(zip(record["option_ids"],
                                                   record["probabilities"]))
                record.pop("option_logits", None)
                record.get("stability", {}).pop("trial_records", None)
                records.append(record)
            results["edge0-logit"] = records

        if "edge0-judgment" in methods:
            records = []
            for index, row in enumerate(rows, start=1):
                print(f"[edge0-judgment] {index}/{len(rows)} {row['id']}",
                      file=sys.stderr)
                option_ids = [o["id"] for o in row["options"]]
                begin = time.perf_counter()
                text = client.chat(judgment_messages(row), max_tokens=700)
                parsed = parse_judgment(text, option_ids)
                records.append({
                    "id": row["id"], "choice": parsed["choice"],
                    "parsed": parsed["parsed"],
                    "forward_seconds": time.perf_counter() - begin,
                    "reasoning": parsed["reasoning"],
                })
            results["edge0-judgment"] = records

        if "comparator" in methods:
            comparator = Comparator(scorer, client.chat,
                                    stability_trials=args.trials)
            records = []
            for index, row in enumerate(rows, start=1):
                print(f"[comparator] {index}/{len(rows)} {row['id']}",
                      file=sys.stderr)
                outcome = comparator.run(row)
                decision = outcome["decision"]
                records.append({
                    "id": row["id"],
                    "choice": decision["choice"],
                    "probabilities": dict(zip(decision["option_ids"],
                                              decision["probabilities"])),
                    "action": outcome["action"],
                    "agreement": outcome["agreement"],
                    "judgment_choice": outcome["judgment"]["choice"],
                    "reasons": outcome["reasons"],
                    "stability": {
                        k: v for k, v in decision.get("stability", {}).items()
                        if k != "trial_records"},
                })
            results["comparator"] = records
    finally:
        server.engine.close()

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": server.model_name,
        # Recorded relative to the working directory when it lies inside
        # it: a report is meant to be shareable, and an absolute path
        # embeds the runner's home directory in every committed artifact.
        "input": _portable_path(args.input),
        "rows": len(rows),
        "stability_trials": args.trials,
        "labels_supplied": bool(labels),
        "wall_seconds": round(time.time() - started, 2),
        "summaries": [summarize(name, records, labels)
                      for name, records in results.items()],
        "records": results,
        "baselines_note": (
            "JEV-CPU 0.6B and JEV MLX 4B figures are not produced here. Run "
            "them with that project's own semif-score CLI on this same input "
            "file and merge, so baseline numbers come from its code."),
        "probability_note": (
            "Reported probabilities are conditional option scores over the "
            "declared options. They are not calibrated real-world "
            "confidence."),
    }
    with open(output, "x") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\nwrote {output}\n")
    for summary in report["summaries"]:
        print(f"--- {summary['method']} ---")
        for key in ("rows", "unparsed", "abstained", "accuracy",
                    "labeled_rows", "mean_top_score", "calibration_gap",
                    "permutation_stable_fraction",
                    "median_probability_spread",
                    "median_forward_seconds"):
            if key in summary and summary[key] is not None:
                value = summary[key]
                print(f"  {key:32s} "
                      f"{value:.4f}" if isinstance(value, float)
                      else f"  {key:32s} {value}")
        if summary.get("accuracy_note"):
            print(f"  note: {summary['accuracy_note']}")
    return 0
