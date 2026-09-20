# JevEdge0 Build Checklist

Authoritative build plan. Any change to these specs requires explicit approval
before implementation.

Source specs: `edg0.md`, `edg1.md`, `jevadd.md`.
Baseline reference: `~/Developer/JEV-CPU` (`src/semif_phase1/{core,direct}.py`).

## Ground rules (from user global instructions)

- No mock, simulation, placeholder, cached response, or demo data anywhere.
  Every number in every report comes from a real forward pass on real weights.
- Each step validated before the next. No partial functionality accepted.
- Any test result below 100% requires a written action plan unless waived.
- Errors logged in `ERRORS.md` paired with mitigation.
- No time/cost/revenue estimates.
- Model versions are selected by the user, never assumed by the assistant.

## Verified preconditions (checked, not assumed)

| Precondition | Status | Evidence |
| --- | --- | --- |
| `Edge0Engine.next_logits()` returns last-position logits | VERIFIED | `src/edge0/engine/base.py` |
| Option letters A–H are single tokens in edge0-35b | VERIFIED | tokenizer round-trip, ids 32–39 |
| Real edge0-35b checkpoint present | VERIFIED | `models/edge0-35b/` 4 shards + LoRA + prerouter |
| MiniLM embedding weights (real) | VERIFIED | downloaded, BERT 6L/384d, 104 tensors |
| pypdf / python-docx | VERIFIED | installed, imported |
| JEV-CPU baseline repo | VERIFIED | cloned to `~/Developer/JEV-CPU` |
| Flask absent; stdlib transport required | VERIFIED | import check |
| Engine lock must be reentrant | VERIFIED | E-006: plain Lock self-deadlocks at /v1/compare |

## Stage 1 — Decision core (`jevedge0/decision/`)

- [x] 1.1 `prompt.py` — port JEV row validation, `LETTERS`, `DIRECT_SYSTEM`,
      JSON payload rendering. Byte-identical prompt contract to the baseline.
- [x] 1.2 `prompt.py` — single-token slot validation + answer-boundary
      invariance check (rejects, never silently truncates).
- [x] 1.3 `scorer.py` — `Edge0DecisionScorer`: reset → prefill → `next_logits()`
      → select slot logits → softmax. No text generation.
- [x] 1.4 `scorer.py` — report entropy, top-two margin, option logits.
- [x] 1.5 `stability.py` — permutation resampling, map back to stable option
      IDs, report winner stability + probability spread.
- [x] 1.6 Validate 1.1–1.5 against real edge0-35b weights.

## Stage 2 — Server surface (`jevedge0/server/`)

- [x] 2.1 `/v1/decisions` on the existing Edge0 process (no second model copy).
- [x] 2.2 `/v1/decision-batch` — several criteria against one state.
- [x] 2.3 Wire into `edge0.server.app` handler dispatch (both transports).
- [x] 2.4 Validate over real HTTP against the loaded model.

## Stage 3 — Comparator (`jevedge0/comparator/`)

- [x] 3.1 Generative-judgment channel (Edge0 chat) + typed-decision channel.
- [x] 3.2 Agreement/disagreement detection across channels.
- [x] 3.3 Guardrail: act / ask / abstain / escalate on thresholds.
- [x] 3.4 Validate on real dual-channel runs.

## Stage 4 — RAG (`jevedge0/rag/`)

- [x] 4.1 Ingestion: PDF, DOCX, MD, HTML, TXT, CSV.
- [x] 4.2 Structure-aware chunking preserving headings/pages/source.
- [x] 4.3 MLX BERT encoder for all-MiniLM-L6-v2 (real weights, mean pooling,
      L2 normalize). Parity-checked against reference embeddings.
- [x] 4.4 Hybrid retrieval: dense cosine + BM25 lexical.
- [x] 4.5 Reranking before context assembly.
- [x] 4.6 Citation enforcement (source + page + quoted span).
- [x] 4.7 Abstention on weak/conflicting evidence.
- [x] 4.8 Prompt-injection defense: documents are untrusted evidence.
- [x] 4.9 Validate retrieval on real documents.

## Stage 5 — Tools (`jevedge0/tools/`)

- [x] 5.1 Structured tool protocol + strict response validation.
- [x] 5.2 Registry with per-tool permission levels from `edg1.md` table.
- [x] 5.3 Initial tools: doc search, read passage, calculator, datetime,
      approved-folder search, approved-file read, SQLite read-only, Python
      sandbox. Shell disabled. Destructive ops require confirmation.
- [x] 5.4 Complete audit log of every invocation.
- [x] 5.5 Validate allow/deny/confirm paths.

## Stage 6 — Store (`jevedge0/store/`)

- [x] 6.1 SQLite: conversations, messages, documents, chunks, audit, memory.
- [x] 6.2 Three-tier memory: history / summary / durable-with-provenance.
- [x] 6.3 Durable memory is proposed, never silent; visible, editable, removable.
- [x] 6.4 Validate persistence across restart.

## Stage 7 — Orchestrator + Web (`jevedge0/orchestrator/`, `jevedge0/web/`)

- [x] 7.1 Agent loop: Edge0 → validate → allowlisted tool → return → repeat.
- [x] 7.2 Malformed output rejected and retried, never executed.
- [x] 7.3 Stdlib HTTP server + SSE (no Flask).
- [x] 7.4 Self-contained vanilla-JS UI (no CDN, works offline): chat, decisions,
      knowledge, memory, audit.
- [ ] 7.5 Validate end to end in a browser. **NOT DONE — BLOCKED.** The
      Chrome extension is not connected to this session, so no browser-driven
      validation was performed and none is claimed. What *was* verified
      programmatically: the UI is served (28,846 bytes, zero external
      resources, balanced tags, all six nav targets resolve to existing
      sections), and every endpoint the page's JS calls returns correctly over
      real HTTP, including the tool-permission PUT and a 404 for unrouted
      paths. Not verified: rendering, click handling, SSE streaming in a real
      browser, and the confirmation dialog.
      **Action:** connect the extension (or open http://127.0.0.1:8090 by hand
      after `jevedge0 serve`) and exercise each tab.

## Stage 8 — Benchmark + CLI

- [x] 8.1 `jevedge0` CLI: decide, batch, ingest, serve, bench.
- [x] 8.2 Benchmark harness comparing, on identical rows:
      JEV-CPU 0.6B · JEV MLX 4B · Edge0-35B logit readout ·
      Edge0 generated judgment · combined comparator.
- [x] 8.3 Report discrimination, calibration, abstention, option-order
      sensitivity, paraphrase stability. Never equate softmax with confidence.
- [x] 8.4 Real side-by-side run; gaps below 100% get an action plan.

## Stage 9 — Tests + docs

- [x] 9.1 Unit tests (no weights) for prompt/validation/softmax/BM25/protocol.
- [x] 9.2 Slow marked tests against real weights.
- [x] 9.3 `ERRORS.md` maintained throughout.
- [x] 9.4 README with verified-only claims; nothing called complete while
      tasks remain.

---

## Validation record

Full-stack run against the real `edge0-35b` checkpoint: **17/17 checks passed**
(`server, decisions, batch, comparator, RAG ingest, hybrid search, grounded
citations, injection resistance, abstention, agent tools, audit, confirm gate,
disabled shell, persistence`).

Test suite: **153 passed, 1 skipped** (pre-existing: flask absent), 11
deselected slow. The 9 slow tests require the checkpoint and run with
`pytest -m slow`.

Benchmark: see `BENCHMARK.md`. Logit readout ~7.6× faster than generation at
equal accuracy on 6 rows — a sample too small to support an accuracy claim,
stated as such there.

### Defects found and fixed during validation

| | Defect | Found by |
| --- | --- | --- |
| E-006 | Engine-lock self-deadlock at `/v1/compare` | Full-stack run (frozen CPU counter) |
| E-007 | Reranker favored short chunks over substantive ones | Reading output of a *passing* check |
| E-008 | No stemming; plurals broke lexical recall | Re-testing E-007 on a different query |

All three carry regression tests. E-007 and E-008 were both found by examining
results that a green assertion had already accepted.
