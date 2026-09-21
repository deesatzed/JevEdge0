# JevEdge0

A private workbench built on Edge0: typed decisions, grounded document
retrieval, controlled tools, and persistent conversations — all running on one
locally loaded model.

JevEdge0 is a layer *above* Edge0, not a fork of it. Edge0 stays the inference
engine; JevEdge0 adds the decision readout, retrieval, tool gateway,
persistence and interface around it.

## Why use this

Edge0 by itself answers one question and stops. JevEdge0 is everything wrapped
around that model to make it an actual assistant: one that remembers
conversations, reads your documents, uses tools, and makes fast typed
decisions instead of writing a paragraph when a paragraph isn't the point.
Think of Edge0 as the engine and JevEdge0 as the car built around it — the
interface, the seatbelts, the tools in the glovebox.

This is aimed at someone who wants a private alternative to a hosted chat
assistant that never sends data anywhere, document Q&A that won't hallucinate
answers from files it never saw, fast automated triage or classification at a
speed full conversational answers can't match, and real tool use (files,
code, data) with an actual safety layer in front of it rather than a
disclaimer.

### The features, and what they're for

**Chat with memory and tools.** Ordinary conversation, but the assistant can
search files, do math, and query your documents mid-conversation rather than
only talking. *Use case:* "Summarize this policy document and calculate what
a 15% increase would cost across these 40 line items" — reads the file, does
the math, answers in one exchange.

**Document search with honest citations.** Documents are indexed and answers
come only from what they actually contain, each claim citing its source
passage. When the documents don't cover the question, it says so instead of
guessing. *Use case:* load your policy manuals, ask a question, get an answer
citing the exact page — not a plausible-sounding fabrication.

**Typed decisions — fast yes/no/multiple-choice.** Rather than writing an
essay and parsing it back out, the model is asked to pick from a fixed list
and the answer comes back in a fraction of the time a full written response
would take. *Use case:* triaging a folder of support tickets into
urgent/normal/can-wait without the model writing an explanation for each one.

**Dual-channel comparator — a built-in second opinion.** For an important
decision, the same question runs two different ways — a quick typed pick and
a fully-reasoned written answer — and the two are compared. Agreement acts;
disagreement flags itself instead of arbitrarily picking one. *Use case:*
auto-approve only when both methods agree; escalate to a human when they
don't.

**Typed risk guardrail.** Before any tool runs, its actual arguments — not
just its name — are classified as harmless, destructive, privileged, or
data-exfiltrating, and risky calls pause for approval even if that tool was
otherwise set to run automatically. *Use case:* a code-writing request that
would delete files or send data out gets flagged for review, automatically,
without you having to think to ask for it.

**Full audit trail.** Every tool action — run, blocked, or failed — is logged
with its risk classification. *Use case:* "what has this assistant actually
done to my files this week?" is a question you can answer, not just trust.

**Memory you control.** The assistant can propose things worth remembering
long-term, but nothing becomes permanent until approved, and anything
remembered can be edited or deleted. Most assistants either forget everything
or silently remember everything; here you choose, every time.

**Guarded code execution.** The assistant can write and run real Python code
to do actual work — data processing, calculations, file manipulation — with
every run requiring confirmation and the exact code shown first. *Use case:*
"parse this CSV and tell me the average" — it writes the code, shows it to
you, you approve, it runs.

**One honest caveat, stated plainly rather than buried:** the fast
risk-classification guardrail tested well on a small sample, but the more
rigorous statistical confidence layer behind it (conformal calibration) is
not yet reliable enough to trust blindly — see [below](#the-typed-risk-guardrail-and-its-conformal-calibration).
That is exactly why it ships disabled, and why risky tool actions still ask
before running rather than deciding silently.

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

**Typed risk guardrail.** Every tool call's actual arguments — not just the
tool's name — are classified as `readonly` / `destructive` / `privileged` /
`exfiltration` before it runs, using the same typed-decision readout as
everything else. A risky classification can only *raise* a call's required
permission level above its static floor, never lower it; a guardrail that
fails, errors, or is disabled leaves the existing static rules unchanged.
Optional split-conformal calibration turns a hand-labeled example set into a
statistically valid uncertainty signal, escalating any call whose prediction
set is ambiguous, independent of the argmax classification.

**Noul and Score.** Beyond the multiple-choice Choice readout, `decision/
primitives.py` adds Noul (a yes/no question with a probability) and an
ordinal Score (a probability-weighted position on an ordered rubric, so 1.8 on
a 0–3 severity scale genuinely means "between levels 1 and 2" — information a
plain argmax discards). Every decision also reports `legal_mass`: how much of
the model's full next-token distribution the declared options actually cover,
a signal for whether the options were ever plausible answers at all.

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

Run everything from the Edge0 repository root with the package on the path:

```bash
export PYTHONPATH="$PWD:$PWD/src"
alias jevedge0="./.venv/bin/python -m jevedge0.cli"
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
jevedge0 bench --input examples/decisions.jsonl \
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
| GET/PUT | `/v1/guard` | Typed risk guardrail settings (enabled, trials, escalation table) |
| GET | `/v1/audit` | Tool audit log (includes each call's guardrail risk verdict) |

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

## The typed risk guardrail, and its conformal calibration

`ToolGuard` classifies each tool call's actual arguments before
`ToolRegistry.invoke()` runs it. The one property that makes this safe to use
even though the classifier is sometimes wrong: it can only *add* restriction.
A `readonly` verdict can never downgrade a tool that already requires
confirmation, and can never re-enable a disabled one; the static permission is
always a floor. Anything that goes wrong in classification — an exception, a
timeout, a malformed decision record, the guard being disabled — leaves the
call under the existing static rules, exactly as if no guard were configured.

Optional split-conformal calibration (`jevedge0/decision/conformal.py`) turns
a hand-labeled example set into prediction *sets* with a statistically valid
coverage guarantee, independent of whether the underlying scores are
calibrated. A non-singleton prediction set — the model is genuinely torn
between two or more risk classes — escalates on its own, regardless of what
the single most-probable class was.

**This calibration is not enabled by default, and here is why, stated
plainly rather than left implicit:** calibrated on 67 real hand-written
examples and measured on 68 disjoint held-out rows, realized coverage came in
at **76.5%** against a 90% target (85% tolerance floor) — a real, reported
shortfall, not a hypothetical caveat. See `BENCHMARK.md` and `ERRORS.md`
(E-013) for the full measurement and diagnosis. The argmax risk
classification itself performed well on the same held-out data (92.6%); the
shortfall is specific to the *conformal coverage guarantee* at this sample
size, which needs substantially more labeled data before it should be relied
on operationally.

## Sandboxing

`run_python` **requires explicit per-call confirmation** — you see the exact
code before it runs, every time, never automatically. It runs in a separate
OS process with a wall-clock timeout, which is real isolation against an
accident — a crash or an infinite loop costs a killed child process, not the
workbench. It is **not a security sandbox** against a deliberate attempt to
escape it: the child process has no network namespace (it can still make
outbound connections), no filesystem jail (it can read or write any path its
OS user can reach, not only the workspace), no memory or CPU ceiling, and no
syscall filter.

Approving a `run_python` call means trusting that specific code as much as you
would trust running it yourself, in your own shell, with your own account's
permissions — read what you're approving. Running genuinely untrusted code
safely needs a VM or container boundary, which this project does not provide;
switching the permission to `automatic`
(`PUT /v1/tools/run_python {"permission": "automatic"}`) removes the
per-call review entirely and is not recommended.

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
