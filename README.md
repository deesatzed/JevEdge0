# JevEdge0

Typed decisions, grounded retrieval, and controlled tools on a locally loaded
[Edge0](https://github.com/Edge0-AI/Edge0) model.

Everything runs on your machine. No API keys, no cloud inference, no telemetry.

## What it does

**Typed decisions.** Instead of asking the model to write prose and then parsing
its answer, JevEdge0 constrains it to a list of options you declare and reads the
model's own next-token scores for the answer letters. One forward pass, no text
generated, and a bounded distribution over exactly the options you offered.

**Permutation stability.** Models are biased toward whichever option appears
first. Every decision can be re-scored under several option orderings and mapped
back to stable option ids, so you can see whether the winner is a property of the
evidence or of the presentation.

**Dual-channel comparison.** The same question runs through the model twice —
once as open generative reasoning, once as the constrained readout. The two
methods fail differently, so their agreement carries more weight than either
alone, and their disagreement flags cases for a human. A guardrail turns the
comparison into an action: act, ask, abstain, or escalate.

**Grounded retrieval.** Documents are ingested with structure intact (pages,
headings, sources), embedded locally, and retrieved with hybrid BM25 + dense
search and reranking. Answers cite passages; citations to passages that were
never retrieved are stripped and reported. When evidence is weak the assistant
abstains and says what is missing.

**Controlled tools.** The model has no native tool calling, so the orchestrator
uses a strict JSON envelope validated before anything runs. Malformed output is
rejected and retried, never guessed at. Every tool carries a permission level,
and every invocation is audited — executed, refused, or failed.

## Requirements

JevEdge0 is a layer above Edge0, not a copy of it. You need:

- Apple Silicon Mac (the Edge0 backend is MLX-only)
- A working [Edge0](https://github.com/Edge0-AI/Edge0) install and an
  `edge0-35b` or `edge0-8b` checkpoint
- Python 3.10+

Model weights are **not** included here — they are tens of gigabytes and belong
to the Edge0 project.

## Install

From inside your Edge0 checkout:

```bash
git clone https://github.com/deesatzed/JevEdge0.git
cd JevEdge0
pip install -e .
pip install pypdf python-docx
```

Point `PYTHONPATH` at both projects, or install Edge0 itself with `pip install -e`.

## Use

```bash
export EDGE0_35B_MODEL=/path/to/models/edge0-35b
jevedge0 serve --allow ~/Documents
```

Open `http://127.0.0.1:8090`. Tabs: Decide, Assistant, Knowledge, Memory, Tools,
Audit.

Only folders passed with `--allow` are readable by the tools; writes are confined
to a workspace under `~/.jevedge0`.

### One decision from the terminal

```bash
jevedge0 decide \
  --state "ED census is 71, 14 admitted boarders, staffing two below plan." \
  --criterion "What is the operational strain over the next four hours?" \
  --option low="Capacity should comfortably exceed demand." \
  --option moderate="Demand may temporarily approach capacity." \
  --option high="Demand is likely to exceed staffed capacity." \
  --option insufficient="The supplied evidence is insufficient." \
  --trials 4
```

### Other commands

```bash
jevedge0 compare --file decision.json --trials 4   # dual channel + verdict
jevedge0 ingest ~/Documents/protocols              # add documents
jevedge0 search "escalation threshold"             # search them
jevedge0 bench --input examples/decisions.jsonl \
  --labels examples/labels.jsonl --output results.json
```

## About the probabilities

The values returned are **conditional option scores**: a softmax over the logits
of the declared option letters. They are not calibrated confidence, and nothing
here treats them as such.

A 0.99 score does not mean a 99% chance of being right. It means that among the
options you offered, the model's next-token distribution concentrated there. If
the correct answer is not among your options, it will still report a high score
for the closest one.

Before attaching operational meaning to these numbers in a given domain,
evaluate discrimination, calibration, abstention, option-order sensitivity, and
paraphrase stability on labeled data from that domain.

## Status and honesty notes

- Validated end to end against real `edge0-35b` weights: 17/17 full-stack checks.
- 153 unit tests passing; 9 further tests require the checkpoint (`pytest -m slow`).
- [`BENCHMARK.md`](jevedge0/BENCHMARK.md) reports a real measured run. The logit
  readout was ~7.6× faster than generation at equal accuracy — on **six rows**,
  a sample far too small to support an accuracy claim, and it says so.
- [`ERRORS.md`](jevedge0/ERRORS.md) logs every defect found during the build,
  including three that were hiding behind passing assertions.
- [`CHECKLIST.md`](jevedge0/CHECKLIST.md) tracks build state, including the one
  item that is **not** done (full browser-driven UI testing).
- The example clinical rows and labels are **synthetic, written as
  illustrations**. They are not real patient data and not an adjudicated gold
  standard. See [`examples/README.md`](jevedge0/examples/README.md).

Not validated for clinical or operational use. Treat it as an experimental
decision primitive.

## Prompt contract and prior work

The decision prompt is byte-identical to the JEV-CPU / SemIf baseline
([leesk212/JEV-CPU](https://github.com/leesk212/JEV-CPU)): same system line,
same JSON payload, same letter alphabet, same single-token round-trip and
answer-boundary validation. That is deliberate — a reworded prompt would
confound a model comparison with a prompt comparison.

JevEdge0 implements that *method* against Edge0's engine. No JEV-CPU code runs
here and no JEV model is loaded.

## Layout

```
jevedge0/
  decision/     prompt contract, logit readout, permutation stability
  comparator/   generative judgment vs typed decision, guardrail
  rag/          ingestion, MLX embeddings, hybrid retrieval, citations
  tools/        envelope protocol, permission registry, built-in tools
  store/        SQLite: chats, documents, memory, decisions, audit
  orchestrator/ agent loop, Edge0 client, memory management
  server/       decision endpoints, stdlib HTTP transport
  web/          single-file browser UI
  bench/        benchmark harness
```

`edg0.md`, `edg1.md` and `jevadd.md` are the design notes this was built from.

## License

Apache-2.0, matching Edge0.
