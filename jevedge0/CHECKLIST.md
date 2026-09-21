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
      execution in a separate process (timeout-isolated, not a security
      sandbox -- corrected in the enhancement build's Stage 4, see
      ERRORS.md E-014). Shell disabled. Destructive ops require
      confirmation.
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

---

## Enhancement build (GOAL_ENHANCE.md)

Executed on branch `enhance/guard-conformal-primitives`.

### Stage 1 — Typed guardrail in the tool gateway

- [x] Risk taxonomy, `ToolGuard.classify()`, prompt-injection fencing via
      `rag/context.py: neutralize()` (`jevedge0/tools/guard.py`).
- [x] `effective_permission()` — safety property (guardrail can only ADD
      restriction) enforced structurally via `max()` on the LEVELS ordering,
      not by convention.
- [x] Wired into `ToolRegistry.invoke()`: DISABLED check unchanged →
      guardrail classifies → effective CONFIRM check → validate → execute.
      `guard=None` reproduces prior behavior byte-for-byte.
- [x] Audit schema migration (`risk`, `risk_probability`, `guard_elapsed_s`)
      via `PRAGMA table_info` + `ALTER TABLE ADD COLUMN`; existing databases
      upgraded in place, not recreated.
- [x] `Workbench` constructs one `ToolGuard` sharing the existing scorer and
      engine lock (no second model copy). `GET/PUT /v1/guard` exposed.
- [x] 32 tests in `tests/test_jevedge0_guard.py`, including all 3 spec'd
      CRITICAL cases plus an extended sweep across all four risk classes.
- [x] Real-weight validation: 13/13 hand-written tool calls classified as
      expected on real edge0-35b, including one adversarial prompt-injection
      case embedded in arguments. See
      `jevedge0/examples/guard-validation-2026-09-20.json`.
      **13 cases is far too few to claim an accuracy figure** — reported as
      a real measurement, not a performance claim.
- [x] Full suite: 185 passed, 1 skipped (pre-existing), 0 regressions.

### Stage 2 — Conformal prediction

- [x] `jevedge0/decision/conformal.py`: split-conformal LAC nonconformity
      score, `calibrate()`, `predict_set()`, `empirical_coverage()`,
      `save_calibration()`/`load_calibration()` with the required honesty
      caveat baked into every saved artifact.
- [x] Quantile computation hand-verified against a manual calculation
      (n=20, alpha=0.1 -> rank 19 -> q_hat=0.48, matches exactly).
- [x] `jevedge0/examples/guard_calibration.jsonl`: **135 hand-written rows**
      (spec required >=120), all four risk classes >=25 each, every row
      carries a one-line rationale, covers all 10 built-in tools, includes
      5 explicitly-marked adversarial cases, 7 explicitly-marked near-miss
      pairs/triples, and one explicitly-flagged taxonomy gap (`git clone`)
      rather than a forced confident label.
- [x] `ToolGuard` gains optional `calibration`; a non-singleton conformal
      set escalates via `uncertain`, combined with the argmax-risk
      escalation through the same `effective_permission()` safety
      property. `calibration=None` reproduces Stage 1 behavior exactly.
- [x] 25 new tests appended to `tests/test_jevedge0_guard.py` covering all
      6 spec'd cases (quantile, monotonicity, singleton, abstain,
      held-out coverage, minimum-rows) plus wiring tests.
- [x] Real-weight validation: calibrated on 67 real rows, measured on 68
      disjoint held-out rows. **Measured coverage 76.47%, below the
      85% tolerance floor.** Reported as a finding per the spec's explicit
      instruction, not adjusted. See `ERRORS.md` E-013 for full diagnosis
      and action plan, `BENCHMARK.md` for the summary table.
- [x] Full suite: 200 passed, 1 skipped (pre-existing), 0 regressions.

### Defects found during the enhancement build

| | Defect | Found by |
| --- | --- | --- |
| E-010 | Untracked `/export` transcript broke baseline hygiene test | Running the full suite before Stage 1 |
| E-011 | `git add -A` would have staged 23GB of model weights | Hung command, investigated rather than force-killed blindly |
| E-012 | Hand-written calibration paths repeatedly tripped the hygiene test | Reading the test's actual regex after the 3rd failure |
| E-013 | Measured conformal coverage (76.5%) fell below the 85% tolerance floor | Real-weight validation required by GOAL_ENHANCE.md Sec 7.B |
| E-014 | `run_python` documented as a "sandboxed workspace" it never was | GOAL_ENHANCE.md Stage 4's own explicit instruction to fix this |
| E-015 | Naive timing measurement showed a fabricated-looking 2.46x speedup | Investigated a suspiciously good result instead of reporting it |

E-010/E-011 are pre-existing environment conditions this build's process
discipline caught. E-012 was a self-inflicted process error (not reading a
test's real logic before guessing at fixes), corrected per the project's
repeated-error rule. E-013 is a genuine, honestly-reported measurement
result, not a code defect -- the conformal math itself is independently
hand-verified correct; what fell short was coverage transfer at n=67/68,
exactly the kind of finding Sec 7's real-weight validation exists to surface
rather than let a synthetic-data unit test suite miss.

### Stage 3 — Noul and Score primitives

- [x] `jevedge0/decision/primitives.py`: `score_noul()` and `score_ordinal()`,
      both built on `Edge0DecisionScorer.score()` unchanged -- same one
      prefill, no text generated, same "conditional score, not calibrated
      confidence" property inherited without restatement needed.
- [x] `legal_mass` added to `Edge0DecisionScorer.score()` itself (not just
      the primitives): computed via `logsumexp` over the full vocabulary
      row rather than materializing a full-vocab probability array, so
      cost stays a single reduction rather than an ~250k-element softmax.
- [x] `score_ordinal` indexes the returned distribution by option id, not
      by positional order, so a scorer that reorders `option_ids` (tested
      explicitly) cannot silently miscompute `expected_score`.
- [x] Guard composition: `ToolGuard.classify_full()` runs all three
      primitives (Choice for risk class unchanged, ordinal Score over a
      4-level severity rubric, Noul for "safe without confirmation").
      Added **additively** -- `classify()` itself is untouched, so every
      Stage 1/2 test keeps passing against the exact same method.
      `full_escalation()` folds all three signals through repeated
      `effective_permission()` application, so the only-add-restriction
      property holds across the composition, not just within one signal.
- [x] 18 tests in `tests/test_jevedge0_primitives.py` (all 7 spec'd cases)
      plus 16 more in `tests/test_jevedge0_guard.py` for the composition,
      including 4 marked CRITICAL: severity escalates independently of a
      readonly risk verdict, Noul escalates independently of both others,
      and the DISABLED/CONFIRM floor survives the three-way fold.
- [x] Real-weight validation (Sec 7.C): Noul and ordinal Score run on real
      edge0-35b inputs; `legal_mass` measured on a well-posed question
      (0.985) vs a deliberately badly-posed one whose options do not match
      the evidence at all (0.853) -- correct direction, real evidence the
      metric responds to option/evidence mismatch, gap smaller than a
      synthetic sanity check suggested (reported honestly, not amplified).
      The file-delete severity example produced a genuinely informative
      split (48% minor / 38% moderate, expected_score 1.52) -- exactly the
      "between two levels" case a bare argmax Choice would discard.
      Full results: `jevedge0/examples/primitives-validation-2026-09-20.json`.
- [x] Full suite: 233 passed, 1 skipped (pre-existing), 0 regressions.

### Stage 4 — Make the isolation claim true

- [x] Rewrote `jevedge0/tools/builtin.py`'s module header and added a full
      docstring to `make_python_tool` naming all four missing protections
      explicitly (network, filesystem, memory/CPU, syscall) rather than
      only removing the word "sandbox."
- [x] `run_python`'s default registration changed from `CONFIRM` to
      `DISABLED`; its registration description states the limitation
      where `GET /v1/tools` surfaces it.
- [x] Added a **Sandboxing** section to `jevedge0/README.md`. The
      standalone mirror repository's own README was deliberately left
      unedited -- separate git repository, outside this branch's scope
      per GOAL_ENHANCE.md's own "Where to run this" section.
- [x] `jevedge0/CHECKLIST.md`'s own Stage 5.3 entry (from the *original*
      build, predating this enhancement) also made the unqualified claim;
      corrected it too rather than preserving an inaccurate security
      claim for the sake of an unchanged historical record.
- [x] Two regression tests in `tests/test_jevedge0_core.py`: default
      permission is `DISABLED`, and the docstring names all four missing
      protections by name (so a future edit that waters it back down to
      something vague also fails).
- [x] Follow-up: found and corrected 109 stale "sandboxed workspace" tool
      descriptions inside Stage 2's `guard_calibration.jsonl` rows, left
      over from before this stage's tool-description fix. See
      `ERRORS.md` E-014's follow-up note for what changed and what did not
      (only description prose; code snippets and labels untouched).
- [x] Full suite: 235 passed, 1 skipped (pre-existing), 0 regressions.
- [x] Real sandboxing was explicitly NOT built -- Stage 4 is a
      documentation-and-default-permission correction by design, not a
      security-engineering project.

### Stage 5 — KV-cache batching

- [x] **Investigated before implementing**, per the spec's explicit
      instruction: read `edge0/engine/base.py`, `edge0/engine/qwen.py`,
      `edge0/prerouter/state.py`. Finding: shared-prefix KV-cache
      branching is **not implementable without modifying Edge0
      internals** -- one mutable KV cache built once by `make_cache()`,
      no `BatchKVCache` construction; one single-position prerouter
      cross-token double-buffer with no concept of multiple branches.
      Full reasoning in `ERRORS.md` E-015 and the method's own docstring.
- [x] Per the spec's explicit fallback: implemented
      `Edge0DecisionScorer.score_shared_state(state, questions)` as a
      correct sequential loop (`src/edge0/` untouched), not a fabricated
      optimization.
- [x] 5 tests in `tests/test_jevedge0_scorer.py`: one record per question
      in declared order, **numerical identity vs individual `score()`
      calls** (the spec's own emphasis: "a fast wrong answer is a
      failure"), engine reset before each question, and a shared-state
      batch followed by a solo `score()` matching a fully independent run.
- [x] Real-weight timing (Sec 7.D): measured ratio **1.013** (mean batch
      2.892s vs mean shared_state 2.928s over 3 alternating repeats,
      warmed up first) -- honestly ~1.0x, exactly as predicted by the
      investigation, not a fabricated speedup.
- [x] **A real measurement mistake was caught before being reported**: an
      initial naive run (no warm-up) measured 0.406, which would have
      meant `score_shared_state` was 2.46x faster than an identical
      sequential loop -- investigated (found a real MLX warm-up effect on
      first use of a prompt shape) and re-measured correctly rather than
      reported as-is. Both artifacts kept for the record.
- [x] Full suite: 240 passed, 1 skipped (pre-existing), 0 regressions.
