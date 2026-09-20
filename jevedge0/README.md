# JevEdge0

A private workbench built on Edge0: typed decisions, grounded document
retrieval, controlled tools, and persistent conversations — all running on one
locally loaded model.

JevEdge0 is a layer *above* Edge0, not a fork of it. Edge0 stays the inference
engine; JevEdge0 adds the decision readout, retrieval, tool gateway,
persistence and interface around it.

## What it does

**Typed decisions.** Edge0 exposes the primitive JEV needs: after prefill,
`next_logits()` returns the full next-token logits. JevEdge0 renders a
constrained prompt, verifies each option letter is exactly one token, reads
only those option-letter logits, and softmaxes across them. One forward pass,
no text generated, a bounded distribution over exactly the options you
declared.

**Stability testing.** A single distribution is sensitive to which letter an
option happened to get. Every decision can be re-scored under several option
orderings and mapped back to stable option ids, so you can see whether the
winner is a property of the evidence or of the presentation.

**Dual-channel comparison.** The same problem runs through Edge0 twice — once
as open generative judgment, once as the constrained logit readout. The two
channels fail differently, so their disagreement is the useful signal. A
guardrail turns the comparison into an action: act, ask, abstain, or escalate.

**Grounded retrieval.** Documents are ingested with their structure intact
(pages, headings, sources), embedded locally, and retrieved with hybrid dense +
BM25 search followed by reranking. Answers cite passages; citations to
passages that were never retrieved are stripped and reported. When the evidence
is weak, the assistant abstains and says what is missing rather than filling
the gap.

**Controlled tools.** Edge0 has no native tool calling, so the orchestrator
uses a strict JSON envelope that is validated before anything runs. Malformed
output is rejected and retried, never guessed at. Every tool has a permission
level, and every invocation is audited whether it ran, was refused, or failed.

**Memory you control.** Durable memory is proposed, never silently committed.
Everything the assistant wants to remember is visible, editable and removable.

## Requirements

- Apple Silicon Mac with an Edge0 checkpoint (`models/edge0-35b`)
- The Edge0 virtual environment
- `pypdf` and `python-docx` for document extraction
- `sentence-transformers/all-MiniLM-L6-v2` weights for embeddings (downloaded
  on first use; the encoder itself runs on MLX, no PyTorch)

No web framework is required. The server is stdlib `http.server` and the UI is
a single self-contained HTML file with no CDN dependencies, so it works
offline.

## Use

Install the package (see the top-level README for full setup), or run from a
checkout with both projects on the path:

```bash
export PYTHONPATH="/path/to/JevEdge0:/path/to/Edge0/src"
export EDGE0_35B_MODEL=/path/to/models/edge0-35b
alias jevedge0="python -m jevedge0.cli"
```

### The workbench

```bash
jevedge0 serve --allow ~/Documents
```

Then open `http://127.0.0.1:8090`. Tabs: Decide, Assistant, Knowledge, Memory,
Tools, Audit.

Only folders passed with `--allow` are readable by the tools. Writes are
confined to the workspace under `~/.jevedge0`.

### A typed decision on the terminal

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

### Dual-channel with a guardrail verdict

```bash
jevedge0 compare --file decision.json --trials 4 --show-reasoning
```

### Documents

```bash
jevedge0 ingest ~/Documents/protocols --collection protocols
jevedge0 search "escalation activation threshold"
```

### Benchmark

```bash
jevedge0 bench --input jevedge0/examples/decisions.jsonl \
  --output results.json --methods edge0-logit,edge0-judgment,comparator \
  --trials 4 --labels labels.jsonl
```

Accuracy is reported only when labels are supplied. The harness does not
produce JEV-CPU baseline figures — run those with that project's own
`semif-score` CLI on the same input file so the numbers come from its code.

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/decisions` | One typed decision |
| POST | `/v1/decision-batch` | Several criteria against one state |
| POST | `/v1/compare` | Dual-channel decision with guardrail verdict |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (Edge0 passthrough) |
| POST | `/v1/agent` · `/v1/agent/stream` | Tool-using agent |
| POST | `/v1/knowledge/ingest` · `/search` · `/ask` | Documents |
| GET/PUT/DELETE | `/v1/memory` | Durable memory |
| GET/PUT | `/v1/tools` | Tools and permissions |
| GET | `/v1/audit` | Tool audit log |

Example:

```bash
curl -s http://127.0.0.1:8090/v1/decisions -H 'Content-Type: application/json' -d '{
  "state": "ED census is 71 with 14 boarders.",
  "criterion": "Is the escalation threshold met?",
  "options": [
    {"id":"activate","description":"The threshold is met."},
    {"id":"monitor","description":"Continue monitoring."},
    {"id":"insufficient","description":"Evidence is insufficient."}
  ],
  "stability_trials": 4
}'
```

## About the probabilities

The values returned are **conditional option scores**: a softmax over the
logits of the declared option letters. They are not calibrated confidence, and
nothing in this project treats them as such.

A 0.95 score does not mean a 95% chance of being right. It means that among the
options you offered, the model's next-token distribution concentrated there.
Before attaching operational meaning to these numbers in a specific domain,
evaluate discrimination, calibration, abstention behavior, option-order
sensitivity, and stability under paraphrasing on labeled data from that domain.

The stability and comparator machinery exists because a single softmax should
not be trusted alone — not as decoration.

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

## Prompt contract

The decision prompt is byte-identical to the JEV-CPU / SemIf baseline
(`semif_phase1/core.py`): same system line, same JSON user payload, same letter
alphabet, same strict single-token round-trip and answer-boundary validation.
That is deliberate — a reworded prompt would confound a model comparison with a
prompt comparison.

## Status

See `CHECKLIST.md` for the build checklist and what has been validated against
real weights, and `ERRORS.md` for the error log.
