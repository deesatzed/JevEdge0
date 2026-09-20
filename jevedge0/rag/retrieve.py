"""Hybrid retrieval: dense vectors + BM25 lexical, then reranking.

Dense search finds paraphrases; lexical search finds the exact drug name,
error code, or patient identifier that a vector model happily blurs into
its neighbours.  Clinical and administrative corpora are full of terms
where the exact string is the point, so neither channel is sufficient
alone and the fusion is not a refinement but a requirement.

Fusion is Reciprocal Rank Fusion, which combines rankings rather than
scores.  BM25 magnitudes and cosine similarities are not commensurable,
so weighting raw scores would silently let one channel dominate depending
on corpus statistics.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

# Common words carry no retrieval signal but do inflate BM25 for long
# chunks. This list is deliberately short: aggressive stoplists drop
# meaningful terms ("no", "not") that invert clinical meaning.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for
with from by as is are was were be been being it its do does did have has
had will would shall should can could may might must
""".split())


def stem(token: str) -> str:
    """Fold the common English inflections lexical matching trips over.

    Not a full stemmer (Porter would be), deliberately: aggressive
    stemming conflates clinically distinct terms, and this corpus is full
    of them. It handles the endings that actually cost recall —
    "threshold" vs "thresholds", "activate" vs "activation" vs
    "activated" — and leaves everything else alone.

    Without this, a query for "escalation threshold" scored zero against
    a passage titled "Activation thresholds" purely on the plural, and a
    document that merely quoted the singular outranked the one that
    defined it.
    """
    if len(token) <= 4:
        return token

    # Verb/noun family: activate / activated / activates / activating /
    # activation all fold to "activat". Applied before the plural rules
    # so "activations" is handled here rather than as a bare plural.
    for suffix in ("ations", "ation", "ating", "ates", "ated", "ate"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[:-len(suffix)] + "at"

    # Plurals. "ies" -> "y"; a bare "s" only when the stem does not
    # already end in s, so "sepsis" and "census" survive intact.
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 5 and token[-3] in "sxzh":
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]

    # Participles: boarding/boarded -> board.
    #
    # Only stripped when the remaining stem still looks like a word, i.e.
    # it contains a vowel and is long enough. Without that guard "exceed"
    # loses a real word ending and becomes "exce", and "staffing" becomes
    # "staf" — both of which stop matching the words they came from.
    for suffix in ("ing", "ed"):
        if not token.endswith(suffix) or len(token) - len(suffix) < 4:
            continue
        base = token[:-len(suffix)]
        if not any(v in base for v in "aeiouy"):
            return token
        # "ed" after a vowel is usually part of the word, not a suffix:
        # exceed, proceed, agreed, need. Leave those alone.
        if suffix == "ed" and base[-1] in "aeiou":
            return token
        # Undo doubling only where English actually doubles before a
        # suffix (plan -> planned). "ff", "ll" and "ss" belong to the
        # base word itself, so "staffing" keeps its second f.
        if (len(base) > 3 and base[-1] == base[-2]
                and base[-1] not in "fls"):
            base = base[:-1]
        # The silent e dropped before the suffix: nursing -> nurs -> nurse,
        # required -> requir -> require. Restored when the stem ends in a
        # consonant cluster or a lone consonant after a vowel.
        if len(base) >= 3 and base[-1] not in "aeiouy" and base[-2] in "aeiouy":
            return base + "e" if _prefers_silent_e(base) else base
        return base
    return token


# Stems whose base form keeps a final e. Restoring it unconditionally
# would corrupt "board" -> "boarde", so the choice is made per stem by
# whether the consonant cluster normally carries one in English.
_SILENT_E_ENDINGS = ("rs", "rc", "rg", "rv", "ss", "nc", "ng", "dg", "lv",
                     "quir", "uir", "clud", "creas", "sur", "vid", "rat",
                     "us", "as", "os", "is")


def _prefers_silent_e(base: str) -> bool:
    return base.endswith(_SILENT_E_ENDINGS)


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower())
            if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over the chunk corpus."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5,
                 b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.length = [len(d) for d in documents]
        self.avg_length = (sum(self.length) / len(self.length)
                           if self.length else 0.0)
        self.frequencies = [Counter(d) for d in documents]
        document_frequency: Counter = Counter()
        for counts in self.frequencies:
            document_frequency.update(counts.keys())
        total = len(documents)
        self.idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        out = np.zeros(len(self.documents), dtype=np.float32)
        if not terms or not self.avg_length:
            return out
        for index, counts in enumerate(self.frequencies):
            length = self.length[index]
            total = 0.0
            for term in terms:
                frequency = counts.get(term)
                if not frequency:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.avg_length)
                total += idf * frequency * (self.k1 + 1) / denominator
            out[index] = total
        return out


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60,
                           weights: list[float] | None = None) -> dict:
    """Fuse ranked index lists into ``{index: score}``."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, index in enumerate(ranking):
            fused[index] = fused.get(index, 0.0) + weight / (k + rank + 1)
    return fused


class Retriever:
    """Hybrid retrieval over a store's chunks.

    The index is built from the store on demand and rebuilt when the
    chunk count changes, so newly ingested documents become searchable
    without restarting the workbench.
    """

    def __init__(self, store, embedder, collection_id: str | None = None):
        self.store = store
        self.embedder = embedder
        self.collection_id = collection_id
        self.chunks: list[dict] = []
        self.matrix: np.ndarray | None = None
        self.bm25: BM25 | None = None
        self.refresh()

    def refresh(self) -> int:
        """Load chunks and rebuild the lexical and dense indexes."""
        self.chunks = self.store.iter_chunks(self.collection_id)
        if not self.chunks:
            self.matrix = None
            self.bm25 = None
            return 0
        vectors = [c["embedding"] for c in self.chunks
                   if c["embedding"] is not None]
        if len(vectors) == len(self.chunks):
            self.matrix = np.vstack(vectors).astype(np.float32)
        else:
            # Partial embedding coverage would make dense scores
            # incomparable across chunks; fall back to lexical only and
            # say so rather than silently ranking on a partial index.
            self.matrix = None
        self.bm25 = BM25([tokenize(c["text"]) for c in self.chunks])
        return len(self.chunks)

    def _ensure_current(self) -> None:
        stored = len(self.store.iter_chunks(self.collection_id))
        if stored != len(self.chunks):
            self.refresh()

    def search(self, query: str, top_k: int = 8,
               candidates: int = 40) -> list[dict]:
        """Retrieve chunks for ``query``, most relevant first."""
        self._ensure_current()
        if not self.chunks:
            return []

        lexical = self.bm25.scores(query)
        lexical_rank = list(np.argsort(-lexical)[:candidates])
        rankings = [lexical_rank]
        weights = [1.0]

        dense = None
        if self.matrix is not None:
            query_vector = self.embedder.encode_one(query)
            dense = self.matrix @ query_vector
            rankings.append(list(np.argsort(-dense)[:candidates]))
            weights.append(1.0)

        fused = reciprocal_rank_fusion(rankings, weights=weights)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])

        results = []
        for index, score in ordered[:candidates]:
            chunk = self.chunks[index]
            results.append({
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "source_path": chunk["source_path"],
                "page": chunk["page"],
                "heading": chunk["heading"],
                "text": chunk["text"],
                "fusion_score": float(score),
                "lexical_score": float(lexical[index]),
                "dense_score": float(dense[index]) if dense is not None else None,
            })
        return self.rerank(query, results)[:top_k]

    # ---- reranking ------------------------------------------------------

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank fused candidates on term coverage and proximity.

        A cross-encoder would rerank better, but it would be a second
        model download and a second forward pass per candidate.  This
        reranker uses signals the fusion stage cannot see: how much of the
        query a chunk actually covers, and whether the matched terms sit
        near each other rather than scattered across unrelated sentences.
        """
        terms = set(tokenize(query))
        if not terms:
            return results
        for result in results:
            tokens = tokenize(result["text"])
            if not tokens:
                result["rerank_score"] = 0.0
                continue
            present = terms & set(tokens)
            coverage = len(present) / len(terms)

            # Density of query terms, normalized against a reference
            # passage length rather than the chunk's own span.
            #
            # An earlier version used raw positional proximity
            # (matched_terms / span). That is a chunk-length artifact,
            # not a relevance signal: a two-line note whose two matches
            # sit adjacent scored a perfect 1.0, while the substantive
            # passage carrying the same terms across real explanatory
            # prose scored 0.12 — so short, uninformative chunks
            # outranked the documents that actually answered the query.
            # Measured on the ED protocol corpus: a staffing note
            # outranked the protocol section defining the thresholds.
            density = min(1.0, len(present) * 120.0 / max(len(tokens), 1))

            # Semantic agreement breaks ties between chunks with equal
            # lexical coverage. This is the signal that knows the
            # protocol section is about escalation and the staffing note
            # merely mentions the word.
            dense = result.get("dense_score")
            semantic = max(0.0, dense) if dense is not None else 0.0

            result["coverage"] = coverage
            result["density"] = density
            result["semantic"] = semantic
            result["rerank_score"] = (
                0.55 * coverage
                + 0.30 * semantic
                + 0.10 * min(1.0, result["fusion_score"] * 20)
                + 0.05 * density)
        results.sort(key=lambda r: -r.get("rerank_score", 0.0))
        return results


def evidence_quality(results: list[dict],
                     min_score: float = 0.25) -> dict:
    """Judge whether retrieved evidence is strong enough to answer on.

    Abstention is a feature: an assistant that always answers from
    whatever came back top-1 will confidently cite the least-bad chunk in
    a corpus that does not contain the answer at all.
    """
    if not results:
        return {"sufficient": False, "reason": "no documents matched the query",
                "top_score": 0.0, "supporting": 0}
    top = results[0].get("rerank_score", 0.0)
    supporting = sum(1 for r in results
                     if r.get("rerank_score", 0.0) >= min_score)
    if top < min_score:
        return {
            "sufficient": False,
            "reason": (f"best match scored {top:.2f}, below the {min_score:.2f} "
                       "evidence threshold"),
            "top_score": top, "supporting": supporting,
        }
    return {"sufficient": True, "reason": "", "top_score": top,
            "supporting": supporting}
