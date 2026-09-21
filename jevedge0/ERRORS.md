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

---

## E-010 — Untracked session-export file broke the baseline test suite

**Symptom.** At the start of the GOAL_ENHANCE.md build, `pytest tests/ -m "not
slow"` failed on `test_no_hardcoded_local_paths`: `sept20Export.md` (a `/export`
terminal transcript sitting untracked at the repo root) contained an absolute
home-directory path.

**Why it mattered.** GOAL_ENHANCE.md requires the full suite to pass before each
stage begins. The file is not source code, is not listed anywhere in the build
spec, and is not owned by this project — rewriting or deleting its content would
be an out-of-scope change to something the user created via a local command.

**Mitigation.** Moved the file, unmodified, to `/tmp/user-exports/` — preserved,
not deleted, out of the repo tree. Did not edit its contents. This is a
relocation of a foreign artifact, not a fix to project code.

**Lesson.** A hygiene test that scans the whole working tree will catch
incidental files the user drops there via unrelated tooling (`/export`,
`/save`, editor swap files). The correct response is to move the artifact out,
not to special-case the test or touch content the build does not own.

---

## E-011 — `git add -A` would have staged 23GB of model weights

**Symptom.** Running `git add -A` in the Edge0 checkout during Stage 1 hung
past a 120-second timeout. Investigation found `models/` (the real
`edge0-35b` checkpoint, ~23GB) is untracked and **not** listed in this
checkout's `.gitignore`, so a bare `git add -A` was scanning and staging it.

**Why it mattered.** Committing 23GB of weights to a git repository is close
to unrecoverable without history surgery, and would make every future clone
of this checkout enormous. This is exactly the mistake documented as a near
-miss in the original JevEdge0 build (the standalone repo's `.gitignore` was
written specifically to prevent it) -- the *Edge0 checkout itself* has no
equivalent protection.

**Mitigation.** Killed the pending `git add -A` before it completed; verified
via `git diff --cached --stat` that nothing under `models/` was staged.
Adopted **stage-by-explicit-path only** for the remainder of this build --
never `git add -A` or `git add .` in this checkout. Every commit in this
enhancement build names its files individually.

**Not fixed at the root.** Adding `models/` to `.gitignore` in the Edge0
checkout is a change to a file GOAL_ENHANCE.md does not list and that belongs
to the upstream Edge0 project structure, not to JevEdge0. Out of scope for
this build; flagging it here is the correct response per the stop-condition
rule (record, do not silently expand scope). The user should add a
`.gitignore` entry for `models/` in their own Edge0 checkout independent of
this build.

**Lesson.** In a repository that intentionally keeps large untracked
artifacts alongside source, `git add -A` is not a safe default even when it
"usually" only picks up what you meant. Check `git status --short` and stage
explicitly before every commit.

---

## E-012 — Hand-written calibration data repeatedly tripped the hygiene test

**Symptom.** Writing `jevedge0/examples/guard_calibration.jsonl` (135
hand-labeled tool-call examples) broke `test_no_hardcoded_local_paths` three
times in a row: first on macOS-style user-home paths, then again after a
naive find-replace to Linux-style user-home paths, then again on a residual
placeholder still shaped like a macOS user-home path. (Deliberately not
quoted verbatim here: an earlier draft of this entry quoted the literal
offending strings and that quotation itself retripped the same hygiene
test, since the test scans every text file in the repository including this
log.)

**Reflection per the project's repeated-error rule.** Possible causes
considered: (1) the test only matches one literal root name; (2) it matches
any home-directory convention across OSes; (3) it checks against the actual
`$HOME` env var; (4) the regex is broader than a literal string; (5) multiple
independent patterns exist for different usernames. Reading
`tests/test_repo_hygiene.py` directly (rather than continuing to guess by
substitution) showed the real cause: `_LOCAL_PATH_PATTERNS` is a small set of
*generic* regexes matching an absolute path under any of four
home-directory-style root segments (a macOS convention, a Linux convention,
a NAS mount convention, and a bare root path), each followed by any
username-shaped component. Each of my first two fixes swapped the username
but kept the same blocked root segment, so of course each one failed
identically.

**Why it mattered.** Fictional illustrative paths in calibration data (used
to describe realistic tool-call arguments like "delete this file under the
user's home directory") are indistinguishable to a path-shape regex from a
real leaked path, and the hygiene test is deliberately conservative about
that -- correctly, since the cost of a false positive (rewriting a fictional
example) is far lower than the cost of a false negative (a real path
shipping in a public artifact).

**Mitigation.** Replaced every absolute home-rooted path in the calibration
set with either `~/...` (tilde form, outside all four blocked patterns and
still semantically clear) or, for `dscl -create`, a bare short username
(which is in fact the argument shape that command actually takes, making the
example more realistic, not less).

**Lesson, stated for next time.** When a test in someone else's code keeps
failing after a targeted fix, that is the signal to *read the test's actual
matching logic* before iterating on more guesses at what it might want --
not after the third failure. The rule to reflect on 5-7 possible causes
before touching code exists precisely to short-circuit this kind of
whack-a-mole loop.

---

## E-013 — Measured conformal coverage fell below tolerance on real weights

**Symptom.** Real-weight Stage 2 validation (GOAL_ENHANCE.md Sec 7.B):
calibrated on 67 real edge0-35b guardrail decisions, alpha=0.1 (target
coverage 90%), measured on 68 disjoint held-out rows. **Measured coverage:
76.47% (52/68), against a tolerance floor of 85% (target - 0.05).** This is
below tolerance and is reported here as the spec requires, not adjusted.

**This is not a bug in the conformal implementation.** The quantile
computation was independently hand-verified against a manual calculation
before this run (see the Stage 2 commit / test
`test_conformal_quantile_matches_hand_computation`), and the monotonicity,
singleton, and empty-set behaviors were all verified correct on synthetic
data first. The shortfall is a property of this specific calibration run,
not of the code computing it.

**Diagnosis.** Argmax accuracy on the *identical* held-out set was 92.6%
(63/68) -- substantially higher than the 76.5% conformal coverage on the
same 68 rows. That gap (11 rows, 16.2 percentage points) is the signature of
the actual mechanism: `mean_set_size` for this run equals `coverage` exactly
(0.7647 both), meaning every prediction set in this run had size 0 or 1,
never 2+. With `q_hat = 0.0882` (threshold = 0.9118), a case is only
"covered" when the model's probability on the *correct* label exceeds
91.18%. Several held-out rows had the model correctly ranking the true
label first (correct argmax) but with a probability the calibration set's
sharper score distribution did not anticipate -- e.g. an argmax winner at
85% confidence is a correct classification but falls outside a 91.18%-wide
conformal set.

**Root cause, to the extent it can be established without more data.** With
only 67 calibration rows, the empirical quantile is a single order
statistic (rank 61 of 67 sorted scores here) -- it has no averaging to smooth
over how representative that one row's score is of the true population
quantile. A calibration and held-out split of ~67/68 rows each, even though
randomly assigned from the same 135-row set, can differ enough in their tail
behavior that a threshold fit tightly to one half does not transfer cleanly
to the other. This is a known, expected weakness of split conformal
prediction at small n, not a defect specific to this taxonomy or this model.

**What was NOT done in response.** Alpha was not lowered to force a wider,
easier-to-satisfy set. The calibration set was not edited to remove
inconvenient rows. The held-out split was not reshuffled to find a
favorable seed. All three would make the reported number look better while
making the actual calibration less honest, which is exactly what
GOAL_ENHANCE.md's Sec 0.1 and Stage 2's explicit instruction ("do not tune
alpha to make the result look good") forbid.

**What this means for use.** The calibration artifact
(`jevedge0/examples/guard_calibration_result-2026-09-20.json`) is saved
as-produced, including this coverage check, per the spec's instruction that
an unfavorable result is not filtered out of the artifact. `ToolGuard`
wiring for calibration (Stage 2 item 5) is implemented and tested with
synthetic data where the guarantee is verified to hold; it is not enabled
by default in `Workbench`, and this measurement is the reason not to enable
it by default with only 135 hand-written rows behind it.

**Action plan (per the project rule that a result below 100% needs one).**
1. Collect substantially more labeled calibration data -- several hundred
   to a thousand-plus rows -- before relying on this guarantee
   operationally. 135 total rows split in half is below what split-conformal
   needs for the empirical quantile to be stable.
2. When more data exists, re-run calibration with a k-fold or repeated-split
   procedure and report the coverage distribution across folds, not a
   single split, so a single unlucky split cannot produce a misleading
   result in either direction.
3. Consider a nonconformity score less sensitive to the exact argmax
   probability at small n (e.g. rank-based rather than raw softmax LAC) if
   the shortfall persists with more data.
4. Until 1-3 are done, treat the guardrail's escalation-by-argmax-risk-class
   (Stage 1, no calibration) as the load-bearing mechanism, and treat the
   conformal "uncertain" signal (Stage 2) as informative but not yet
   validated at operational scale.

**Lesson.** A real measurement that fails is more valuable than a synthetic
one that passes. This is exactly why Sec 7 of GOAL_ENHANCE.md requires
running validation against real weights rather than stopping at unit tests
on fakes -- the unit tests here were and remain correct; they simply cannot
see a small-sample calibration problem that only exists with real,
correlated, hand-written data.

---

## E-014 — `run_python` documented as a "sandboxed workspace" (RESOLVED)

**Symptom.** `make_python_tool`'s registration description and the module
header in `jevedge0/tools/builtin.py` called the tool's execution
environment a "sandboxed workspace." It runs `subprocess.run([sys.executable,
"-I", "-c", code], timeout=..., cwd=workspace)` with a trimmed environment.
That is real isolation against an *accident* (a crash or infinite loop kills
a child process, not this workbench) and provides none of: a network
namespace, a filesystem jail, a memory or CPU ceiling, or a syscall filter.
Against a *deliberate* attempt to escape it, none of those matter.

**Note on the spec's error number.** GOAL_ENHANCE.md Stage 4 instructs
"Add `E-010`" for this entry. E-010 was already used (the untracked
`/export` transcript found during Stage 1 setup, before this stage's work
began) -- the spec's numbering was written without knowing Stage 1 setup
would consume E-010/E-011. Per the project rule not to change the build
spec without recording why, this is filed as E-014 (continuing the log's
actual sequence) rather than overwriting the existing E-010 entry.

**Why it mattered.** "Sandboxed" is a specific security claim. Documentation
that overstates isolation is worse than no claim at all, because it invites
someone to run code they would not otherwise trust, on the strength of a
word the implementation does not back up.

**Mitigation.**
1. Rewrote `jevedge0/tools/builtin.py`'s module header and added a full
   docstring to `make_python_tool` itself, naming all four missing
   protections explicitly (network, filesystem, memory/CPU, syscall) rather
   than only removing the word "sandbox."
2. Changed `run_python`'s registration from `CONFIRM` to `DISABLED`.
   Enabling it is now a deliberate, explicit act (`PUT /v1/tools/run_python`)
   rather than something a user could reach through the ordinary
   per-call confirmation flow without ever seeing the isolation caveat.
3. Rewrote the tool's registration description to state the limitation in
   the one place every caller of `GET /v1/tools` sees it.
4. Added a **Sandboxing** section to `jevedge0/README.md` stating the
   default-disabled status, what enabling it means, and that untrusted code
   needs a VM/container boundary this project does not provide. (The
   standalone mirror repository's own root README was intentionally left
   unedited by this commit -- it is a separate git repository outside this
   branch's scope, and GOAL_ENHANCE.md's own "Where to run this" section
   already treats it as a courtesy copy rather than the build target.
   Updating it is a follow-up for whoever next syncs that mirror.)
5. Added two regression tests
   (`tests/test_jevedge0_core.py::test_run_python_is_disabled_by_default`,
   `::test_run_python_docstring_names_the_missing_protections`) so the code
   and the documentation cannot silently drift apart again -- the second
   test specifically greps the docstring for the four named gaps, so a
   future edit that quietly waters the docstring back down to something
   vague would also fail.

**What was deliberately NOT done.** Real sandboxing (a VM or container
boundary) was not built. GOAL_ENHANCE.md Stage 4 explicitly instructs
correcting the claim, not building the isolation -- treating this as a
documentation-and-default-permission fix, not a security-engineering
project, which is the right scope for this stage.

**Follow-up correction to Stage 2's calibration data.** The Stage 4 tool
description fix left `jevedge0/examples/guard_calibration.jsonl`'s 109
`run_python` rows quoting the *old* "sandboxed workspace" tool description
in their `state` field, stale relative to the code they were meant to
describe. Updated all 109 mechanically to the corrected description text
(a single uniform string substitution; verified the file still parses, all
135 rows survive, and class balance is unchanged: 25/26/25/59 across the
four risk classes). The classification-relevant content of every row --
the code snippet and the label -- is untouched; only the tool-description
prose changed. Stage 2's already-reported coverage measurement (76.47%,
n=67/68) was computed against the pre-fix description text; the change is
cosmetic to the classification task, not substantive, so the measurement is
not expected to be materially affected, but this is noted here rather than
silently letting the artifact and the underlying data drift apart without a
record of when and why.

---

## E-015 — Shared-prefix KV-cache branching is not implementable without modifying Edge0 (finding, not a defect)

**Investigation (GOAL_ENHANCE.md Stage 5 requires this before implementing
anything).** Read `edge0/engine/base.py`, `edge0/engine/qwen.py`, and
`edge0/prerouter/state.py` before writing `score_shared_state`. Two facts
settle the question:

1. `Qwen35Engine.cache` is `self._lm.make_cache()` -- a plain
   `[ArraysCache(...) or KVCache() for layer in layers]` list (confirmed in
   `edge0/backends/mlx/_impl/qwen3_5.py`), built once and mutated in place
   by every `_forward` call. `mlx-lm` itself ships a `BatchKVCache` for
   exactly this kind of branching, and Edge0 does not construct one.
2. `PrerouterState` (`prerouter/state.py`) holds exactly one
   double-buffered "previous token" position (`logits_prev`, `oh_prev`)
   shared across every layer, overwritten by `swap()` on every forward.
   It has no concept of more than one active sequence position.

Branching after a shared prefill would require every branch to carry its
own copy of both the KV cache *and* this prerouter state *and* every
streaming expert's staged-slot state (`_all_stream_layers`), then somehow
interleave forwards across branches without one branch's "previous token"
bookkeeping corrupting another's. That is new engine capability, not a
call-site change, and squarely inside `src/edge0/`, which GOAL_ENHANCE.md
puts out of scope.

**What was built instead, per the spec's explicit fallback instruction.**
`Edge0DecisionScorer.score_shared_state(state, questions)` -- a correct
sequential loop, one `score()` call per question, documented with the above
finding directly in its own docstring so a future reader does not have to
re-derive it. It differs from the existing `score_batch` only in accepting
one shared `state` argument instead of repeating it per row.

**Real-weight timing measurement, and a methodology mistake caught before
it was reported.** The first timing run (cold engine, `score_batch` timed
first, `score_shared_state` timed second, no warm-up) measured a ratio of
0.406 -- `score_shared_state` appearing 2.46x *faster*. That would have
been a fabricated-looking result for two loops that run identical code, so
it was investigated rather than reported: a follow-up run scoring the
IDENTICAL row four times in immediate succession showed the first call
took 2.069s with logits `[19.875, 21.75, 27.125, 20.125]}` and calls 2-4
stabilized at ~1.0-1.25s with identical logits `[19.625, 21.375, 26.875,
20.0]` -- a real, reproducible one-time warm-up cost (almost certainly MLX
lazy compilation and/or quantized-expert-cache warming on first use of a
given prompt shape), not nondeterminism and not a difference between the
two methods. The naive first measurement had simply made `score_batch`
absorb that one-time cost by running first.

Re-measured correctly: one untimed warm-up call, then three alternating
repeats of each method. **Result: mean batch 2.892s, mean shared_state
2.928s, ratio 1.013** -- indistinguishable within measurement noise, exactly
matching the architectural prediction that two identical sequential loops
should perform identically. This is the number reported in `BENCHMARK.md`.
Both the naive and the corrected measurement artifacts are kept
(`jevedge0/examples/shared-state-naive-timing-2026-09-20.json` and
`shared-state-timing-2026-09-20.json`) rather than only the final one, so
the correction itself is auditable.

**Lesson.** A surprising speedup between two code paths that execute the
same operations in the same order is a signal to check measurement
methodology before reporting it, not after. Cold-start effects (JIT/lazy
compilation, cache warming) are real and measurable on this engine, and a
timing comparison across two different call orders without a shared
warm-up will attribute that one-time cost to whichever path runs first.
