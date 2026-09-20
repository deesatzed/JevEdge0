"""Knowledge service: ingest documents, embed chunks, answer with citations."""

from __future__ import annotations

import os

from jevedge0.rag.context import (abstention_message, build_messages,
                                  check_citations)
from jevedge0.rag.embed import get_embedder
from jevedge0.rag.ingest import SUPPORTED, ingest_file
from jevedge0.rag.retrieve import Retriever, evidence_quality


class KnowledgeService:
    """Document ingestion and grounded retrieval over a store."""

    def __init__(self, store, embedder=None):
        self.store = store
        self.embedder = embedder or get_embedder()
        self._retrievers: dict[str | None, Retriever] = {}

    def retriever(self, collection_id: str | None = None) -> Retriever:
        if collection_id not in self._retrievers:
            self._retrievers[collection_id] = Retriever(
                self.store, self.embedder, collection_id)
        return self._retrievers[collection_id]

    # ---- ingestion ------------------------------------------------------

    def ingest(self, path: str, collection: str = "default") -> dict:
        """Ingest one file into a collection."""
        collection_id = self.store.create_collection(collection)
        parsed = ingest_file(path)
        document_id = self.store.add_document(
            collection_id=collection_id,
            source_path=parsed["source_path"],
            filename=parsed["filename"],
            media_type=parsed["media_type"],
            sha256=parsed["sha256"],
            page_count=parsed["page_count"],
        )
        if document_id is None:
            return {"status": "duplicate", "filename": parsed["filename"],
                    "chunks": 0,
                    "detail": "identical content already in this collection"}

        chunks = parsed["chunks"]
        vectors = self.embedder.encode([c["text"] for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        stored = self.store.add_chunks(document_id, chunks)
        self._retrievers.clear()
        return {
            "status": "ingested",
            "document_id": document_id,
            "filename": parsed["filename"],
            "media_type": parsed["media_type"],
            "pages": parsed["page_count"],
            "chunks": stored,
            "collection": collection,
        }

    def ingest_folder(self, folder: str, collection: str = "default",
                      recursive: bool = True) -> dict:
        """Ingest every supported file under a folder."""
        if not os.path.isdir(folder):
            raise NotADirectoryError(folder)
        results, failures = [], []
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() not in SUPPORTED:
                    continue
                path = os.path.join(root, name)
                try:
                    results.append(self.ingest(path, collection))
                except Exception as exc:  # noqa: BLE001 - report, continue
                    failures.append({"file": path, "error": str(exc)})
            if not recursive:
                break
        return {
            "ingested": sum(1 for r in results if r["status"] == "ingested"),
            "duplicates": sum(1 for r in results if r["status"] == "duplicate"),
            "chunks": sum(r["chunks"] for r in results),
            "failed": failures,
            "results": results,
        }

    # ---- retrieval ------------------------------------------------------

    def search(self, query: str, top_k: int = 8,
               collection: str | None = None) -> list[dict]:
        collection_id = self._collection_id(collection)
        return self.retriever(collection_id).search(query, top_k=top_k)

    def _collection_id(self, collection: str | None) -> str | None:
        if not collection:
            return None
        for row in self.store.list_collections():
            if row["name"] == collection:
                return row["id"]
        return None

    def answer(self, question: str, chat_fn, top_k: int = 6,
               collection: str | None = None,
               history: list[dict] | None = None,
               min_score: float = 0.25) -> dict:
        """Answer a question from documents, or abstain with a reason.

        ``chat_fn(messages) -> str`` runs the generation; the service
        stays independent of how Edge0 is reached.
        """
        results = self.search(question, top_k=top_k, collection=collection)
        quality = evidence_quality(results, min_score=min_score)
        if not quality["sufficient"]:
            return {
                "answered": False,
                "abstained": True,
                "answer": abstention_message(quality, question),
                "citations": [],
                "evidence_quality": quality,
                "retrieved": results,
            }
        messages, citations = build_messages(question, results, history=history)
        raw = chat_fn(messages)
        checked = check_citations(raw, citations)
        return {
            "answered": True,
            "abstained": False,
            "answer": checked["answer"],
            "citations": checked["used_citations"],
            "invalid_markers": checked["invalid_markers"],
            "grounded": checked["grounded"],
            "uncited": checked["uncited"],
            "evidence_quality": quality,
            "retrieved": results,
        }
