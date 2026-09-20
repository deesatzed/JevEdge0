# Benchmark: measured results

Real run, real weights, no estimates. Raw output:
`examples/benchmark-run-2026-09-19.json`.

```
jevedge0 bench --input jevedge0/examples/decisions.jsonl \
  --labels jevedge0/examples/labels.jsonl \
  --methods edge0-logit,edge0-judgment,comparator --trials 4
```

Model `edge0-35b`, 6 rows, 4 option orderings per decision, 150.6s wall.

## Results

| | logit readout | generated judgment | comparator |
| --- | --- | --- | --- |
| Accuracy (6 labeled rows) | 6/6 | 6/6 | 6/6 |
| Unparseable responses | 0 | 0 | 0 |
| Median forward time | **1.10 s** | 8.37 s | — |
| Permutation stability | 1.00 | n/a | 1.00 |
| Median probability spread | 0.0044 | n/a | 0.0049 |
| Mean top score | 0.969 | n/a | 0.972 |
| Calibration gap | −0.031 | n/a | −0.028 |

Per row, both channels agreed everywhere, so the guardrail returned ACT on
all six:

| row | logit | judgment | agree | verdict |
| --- | --- | --- | --- | --- |
| ed-strain-1 | high | high | yes | ACT |
| ed-escalate-1 | activate | activate | yes | ACT |
| ed-thin-1 | insufficient | insufficient | yes | ACT |
| drift-1 | drift | drift | yes | ACT |
| deploy-1 | yes | yes | yes | ACT |
| route-1 | account_access | account_access | yes | ACT |

## What this does and does not establish

**Supported by the measurement.**

The logit readout answered these rows about **7.6× faster** than generation
(1.10 s vs 8.37 s median) at the same accuracy. That ratio is the real
argument for the typed path: it is one prefill instead of a generation loop.

Permutation stability was perfect and probability spread was ~0.004, so on
these rows the winner was a property of the evidence rather than of which
letter an option received. That is the specific failure mode the stability
machinery exists to detect, and it did not occur here.

The `ed-thin-1` row — deliberately starved of evidence — was answered
`insufficient` by both channels rather than guessed. Abstention worked.

**Not supported by the measurement.**

Six rows is far too few to establish accuracy. 6/6 is consistent with a true
accuracy anywhere from roughly 60% upward; the 95% confidence interval on
6/6 successes runs from about 0.61 to 1.00. Treat the accuracy column as
"nothing broke", not as a performance claim.

The labels are reasoned expectations written while building this project, not
an adjudicated gold standard (see `examples/README.md`). Accuracy against them
measures agreement with the author, not clinical correctness.

The calibration gap (−0.03) is computed over a single populated bin, since
every decision landed above 0.8. One bin cannot show a calibration curve. It
does not demonstrate that the scores are calibrated; it shows they were
confident and, on this tiny set, right.

Because both channels agreed on every row, this run exercised the ACT path
only. The ESCALATE, ABSTAIN and ASK paths were verified separately by unit
test, not by this benchmark.

**Baselines are absent, not omitted.**

JEV-CPU 0.6B and JEV MLX 4B figures are not here. JEV-CPU's loader requires
exactly one visible CUDA GPU (`semif_phase1/core.py: load_causal_model`), so
its PyTorch path cannot run on this Mac at all; its MLX path needs a separate
environment and a ~9 GB download. Producing those numbers requires running
that project's own `semif-score` CLI on `examples/decisions.jsonl` and merging
the result. No baseline figures were estimated or filled in.

## To make this a real evaluation

1. Build a labeled set from the target domain, with real adjudication,
   at a size where accuracy has a usable confidence interval.
2. Include rows where the correct answer is `insufficient`, and rows where
   the two channels should disagree, so the guardrail's other paths are
   measured rather than assumed.
3. Spread scores across the probability range so calibration has more than
   one populated bin.
4. Run the JEV-CPU baselines on identical rows through their own CLI.
5. Test paraphrase stability (`paraphrase_stability`) with caller-supplied
   rewordings, which this run did not exercise.
