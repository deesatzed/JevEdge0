# JevEdge0 Error Log

Errors encountered during the build, paired with the mitigation actually
applied. Kept per the project rule that every error carries its fix or its
transition.

---

## E-001 — Cached embedding model was not actually present

**Symptom.** `~/.cache/huggingface/hub` listed
`models--sentence-transformers--all-MiniLM-L6-v2`, which read as "the model is
already local". It was not: the directory contained only `.no_exist/` negative
cache markers from an earlier probe. No weights, no config.

**Why it mattered.** The RAG design was approved on the understanding that no
embedding download was needed. Building on that assumption would have produced
an encoder that failed at first use, or worse, a silent fallback.

**Mitigation.** Verified the directory contents before relying on them, then
downloaded the real weights (`model.safetensors`, 104 tensors, BERT 6L/384d).
Confirmed present on the external volume the user's `HF_HOME` points at.

**Rule reinforced.** A cache directory's existence is not evidence its contents
exist. Check for the weight file, not the folder.

---

## E-002 — No reference implementation available to parity-check embeddings

**Symptom.** The MLX BERT encoder is a from-scratch implementation. With
`sentence-transformers` deliberately not installed, there is no reference
implementation on this machine to diff vectors against.

**Why it mattered.** A wrong weight transpose, a missed LayerNorm, or tanh-gelu
instead of erf-gelu all produce an encoder that runs and returns plausible
384-dim vectors while retrieving badly. That failure is silent.

**Mitigation.** Replaced exact-vector parity with behavioral verification
(`BertEmbedder.verify_parity`) asserting the properties a correct encoder must
have: unit norm after L2 normalization, related sentences scoring far above
unrelated ones, and batch-vs-single invariance.

**Measured result.** dimension 384, unit_norm True, related 0.508, unrelated
0.047, batch drift 9.1e-08. A transposed or mis-normalized encoder fails these.

**Residual risk.** Behavioral checks cannot prove bit-exact agreement with the
reference implementation. If exact parity is ever required, install
`sentence-transformers` in a throwaway environment and diff vectors directly.

---

## E-003 — `next_logits()` shape is not fixed across engine paths

**Symptom.** `Edge0Engine._forward` returns logits whose rank depends on the
family path taken; indexing `logits[slot]` directly would silently read the
wrong axis on one of them.

**Why it mattered.** Reading the wrong axis yields numbers that softmax
normally and look like a decision. There is no exception, only a wrong answer.

**Mitigation.** `Edge0DecisionScorer._select` reduces to the final position's
vocabulary row (`while ndim > 1: row = row[-1]`) before indexing slots, and
calls `core.eval` before `.item()`.

**Verified.** Real 35B run produced a valid distribution summing to 1.0 with
the expected winner, stable across permutations.

---

## E-004 — Baseline comparison cannot be fabricated

**Symptom.** The benchmark spec asks for a five-way comparison including
JEV-CPU 0.6B and JEV MLX 4B. Those require a separate environment (JEV-CPU's
own venv) and, for the 4B path, a ~9 GB download. The JEV-CPU loader also
requires CUDA (`load_causal_model` raises without exactly one visible CUDA
GPU), so its PyTorch path will not run on this Mac at all.

**Why it mattered.** Reporting baseline numbers this machine did not produce
would be fabricated data.

**Mitigation.** The harness computes only what it actually measures (the Edge0
methods) and carries an explicit `baselines_note` stating that JEV-CPU figures
must come from that project's own `semif-score` CLI on the same input file.
No placeholder rows, no estimated baselines.

**Status.** JEV-CPU cloned to `~/Developer/JEV-CPU`. Its MLX backend is the
runnable path on Apple Silicon; its CUDA-gated PyTorch path is not.

---

## E-005 — CSS typo in the dark-theme token block

**Symptom.** `--ink:#ecece f` (stray space) in `web/ui.html`, in the
`:root[data-theme="dark"]` block.

**Why it mattered.** An invalid value makes the declaration dropped, so an
explicit dark-theme selection would inherit the wrong text color.

**Mitigation.** Corrected to `--ink:#ececef`. Caught on re-read before the UI
was served.

---

## E-006 — Self-deadlock on the engine lock at `/v1/compare` (RESOLVED)

**Symptom.** Full-stack validation passed six checks, then hung permanently at
`POST /v1/compare`. The process sat at 0% CPU with 15 GB resident and a frozen
CPU-time counter (10.47s, unchanging across sampling) while holding an
established connection on the server port. Not slow — stopped.

**Root cause.** Edge0's `QueueServer` guards its single engine slot with a
plain `threading.Lock`. The comparator route acquired that lock and then, inside
the hold, ran the generative channel via `QueueServer.chat`, which tries to
acquire the *same non-reentrant lock on the same thread*. Guaranteed deadlock,
100% reproducible on the first `/v1/compare` request.

A plain `Lock` is correct for Edge0, where one request is one generation. It is
wrong for JevEdge0, where a single logical operation deliberately spans a
decision and a generation that must not be interleaved with other work.

**Why the unit tests missed it.** The comparator's control flow was tested with
a scripted client that never touched a real `QueueServer`, so the nested
acquisition never happened. The deadlock lived in the *composition* of
comparator and server, which only the full-stack run exercised.

**Mitigation.** Added `jevedge0.server.decisions.engine_lock`, which promotes
the server's lock to a `threading.RLock` once. Reentrancy lets one thread hold
the engine across several operations meant to be atomic; mutual exclusion
across threads is unchanged. `DecisionService` and `Workbench` share that one
lock object.

**Regression tests added** (`tests/test_jevedge0_core.py`):
- `test_engine_lock_is_reentrant` — nested acquire succeeds (fails on a plain
  `Lock`, which is exactly the hang).
- `test_engine_lock_still_excludes_other_threads` — a second thread is still
  blocked, proving reentrancy did not weaken exclusion.
- `test_engine_lock_is_promoted_only_once` — promotion is idempotent.

**Lesson.** Deadlocks live in composition, not in components. Any layer that
holds a resource and then calls back into the layer that owns it needs a
reentrant lock or an explicit unlocked entry point.

---

## E-007 — Reranker systematically favored short chunks (RESOLVED)

**Symptom.** On the real ED protocol corpus, the query "escalation activation
threshold boarding" ranked `injection_test.md` (a two-line staffing note)
*above* the `ed_escalation.md` passage that actually defines the thresholds —
despite the note having lower semantic similarity (dense 0.490 vs 0.563).

**How it was found.** Not by a failing test. The validation check asserted only
that *some* result came back containing "escalation", which passed. The defect
was visible in the reported top-1 filename and was caught by reading the
result rather than trusting the green check.

**Root cause.** The reranker's proximity term was
`matched_terms / positional_span`. That measures chunk brevity, not relevance:
two matched terms sitting adjacent in a 5-token note scored a perfect 1.00,
while the same terms spread across a 43-token explanatory passage scored 0.12.
Every substantive passage was penalized for being substantive.

**Why it mattered beyond ranking.** The corpus contained a prompt-injection
document. A reranker biased toward short, terse chunks preferentially surfaces
exactly the shape that injected instructions take. The retrieval bug and the
security surface pointed the same direction.

**Mitigation.** Replaced raw proximity with two signals that are not
length artifacts:
- `density` — matched terms per reference passage length, capped at 1.0, and
  weighted at only 0.05.
- `semantic` — the dense cosine score, weighted 0.30, breaking ties between
  chunks of equal lexical coverage using the signal that actually models
  meaning.

New weights: coverage 0.55, semantic 0.30, fusion 0.10, density 0.05.

**Measured result.** The protocol passage now ranks first on every probe query
("escalation activation threshold boarding", "when should escalation be
activated", "what happens on activation", "distribution drift monitoring PSI").

**Regression tests added** (`tests/test_jevedge0_core.py`):
- `test_rerank_does_not_favor_short_chunks_over_substantive_ones` — equal
  lexical coverage, terse vs substantive; the substantive chunk must win.
- `test_rerank_uses_semantic_score_as_tiebreaker`.

**Note on the first test fixture.** The initial version of that test failed,
and the failure was in the *fixture*, not the code: the passage said
"activated" (a different token from the query's "activation") and omitted
"threshold", so its lower coverage was correct. Rewritten so both chunks match
the same terms, isolating the length bias the test is meant to catch. Worth
recording: a red test is not automatically a real defect.

**Lesson.** A passing assertion is not a correct result. `assert "escalation"
in top_result` passed while the ranking was wrong; only reading the actual
output revealed it.

---

## E-008 — Lexical channel had no stemming; plurals broke recall (RESOLVED)

**Symptom.** After fixing E-007, the query "escalation threshold" *still*
ranked the injection document first. The E-007 fix was verified on a longer
4-term query and I generalized from that single probe — the shorter query was
never retested until browser-endpoint validation surfaced it.

**Root cause, and why it was not the reranker.** The injection note literally
contains "escalation threshold" (from its injected text "the escalation
threshold is 200 patients"), so its coverage of 1.00 was *correct*. The real
protocol passages say "thresholds" — plural — which `tokenize` treated as a
completely unrelated term, giving them coverage 0.50. The reranker was
faithfully ranking bad input.

The defect was in tokenization: no stemming at all, so "threshold"/"thresholds"
and "activate"/"activation"/"activated" were unrelated tokens throughout BM25
and coverage scoring. Every inflected query silently lost recall.

**Security relevance.** A document quoting a term verbatim outranked the
document defining it. For a corpus containing injected content, verbatim
quoting is exactly the adversary's advantage.

**Mitigation.** Added `jevedge0.rag.retrieve.stem`, a conservative stemmer
folding the inflections that actually cost recall (plurals, `-ation`/`-ate`
verb families, `-ing`/`-ed` participles) while explicitly protecting words
where over-stemming would conflate distinct clinical terms.

**Iterations before it was correct** (each caught by running it, not by
assuming):
- v1 mangled real words: `exceed`→`exce`, `sepsis`→`sepsi`, `staffing`→`staf`.
- v2 fixed the `-ed`-after-vowel case (`exceed`, `need`, `agreed`) and the
  silent-e restoration (`nursing`→`nurse`, `required`→`require`).
- v3 stopped de-doubling `ff`/`ll`/`ss`, which belong to the base word
  (`staffing`→`staff`, not `staf`).

**Measured result.** 13/15 inflection pairs fold correctly; the full integrity
set (`sepsis`, `census`, `analysis`, `status`, `diagnosis`, `process`, `still`,
`need`, `board`, `staff`, `drift`, `psi`, `less`) is unchanged. `nurse/nursing`
and `arrive/arrival` remain unmatched — irregular forms a full Porter stemmer
would need. Left deliberately: chasing them risks corrupting clinical
vocabulary for marginal recall.

After re-ingesting with the stemmed tokenizer, the injection document no longer
wins *any* probe query; every one returns the correct protocol passage.

**Regression tests added** (`tests/test_jevedge0_core.py`): 14 inflection-pair
cases, 14 integrity cases, `test_tokenize_applies_stemming_and_stopwords`, and
`test_bm25_matches_across_inflection`.

**Note.** Stemming changes stored token statistics, so an index built before
this fix must be re-ingested to benefit.

**Lesson.** Verifying a fix on one input is not verifying the fix. E-007's
repair was real but incomplete, and the single probe I used to confirm it
happened to be the one that masked the remaining defect.

---

## E-009 — Benchmark report embedded an absolute home path (RESOLVED)

**Symptom.** `tests/test_repo_hygiene.py::test_no_hardcoded_local_paths` failed
after the benchmark artifact was added: the report's `input` field carried
`/Users/<user>/Developer/...`.

**Why it mattered.** Edge0's hygiene test exists because committed artifacts
carrying a developer's home directory leak the username and break for everyone
else. The benchmark report is meant to be shareable, so it is exactly the wrong
place for an absolute path.

**Mitigation.** Added `_portable_path` to the benchmark runner: paths inside
the working directory are recorded relative, paths under the home directory are
`~`-prefixed, and anything else stays absolute. Fixed at the source rather than
only in the committed file, so future runs are clean too. The existing artifact
was sanitized to match.

**Credit where due.** This was caught by Edge0's own pre-existing hygiene test,
not by anything added here — a good argument for running the whole suite rather
than only one's own tests.
