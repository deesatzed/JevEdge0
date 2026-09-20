"""SQLite persistence for the JevEdge0 workbench.

One file holds conversations, messages, documents, chunks, embeddings,
durable memory, decision records and the tool audit log.  SQLite is the
right size for a single-user local workbench: no server, transactional,
and inspectable with any sqlite3 client if the user wants to audit what
the assistant stored about them.

Embeddings live in a BLOB column as raw float32.  A dedicated vector
store would matter at a scale this workbench does not target; a linear
scan over a local corpus is fast and keeps the dependency count at zero.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New conversation',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    summary     TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    reasoning        TEXT NOT NULL DEFAULT '',
    citations        TEXT NOT NULL DEFAULT '[]',
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv
    ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id             TEXT PRIMARY KEY,
    collection_id  TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    source_path    TEXT NOT NULL,
    filename       TEXT NOT NULL,
    media_type     TEXT NOT NULL,
    sha256         TEXT NOT NULL,
    page_count     INTEGER NOT NULL DEFAULT 0,
    metadata       TEXT NOT NULL DEFAULT '{}',
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_collection
    ON documents(collection_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha
    ON documents(collection_id, sha256);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    page         INTEGER,
    heading      TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL DEFAULT 0,
    embedding    BLOB,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);

CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    content     TEXT NOT NULL,
    provenance  TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'proposed',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);

CREATE TABLE IF NOT EXISTS decisions (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT,
    row_id           TEXT NOT NULL,
    state            TEXT NOT NULL,
    criterion        TEXT NOT NULL,
    options          TEXT NOT NULL,
    record           TEXT NOT NULL,
    created_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audit (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT,
    tool             TEXT NOT NULL,
    arguments        TEXT NOT NULL,
    decision         TEXT NOT NULL,
    result_summary   TEXT NOT NULL DEFAULT '',
    error            TEXT NOT NULL DEFAULT '',
    duration_s       REAL NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit(created_at);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def pack_vector(vector) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    """Thread-safe SQLite store."""

    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ---- conversations -------------------------------------------------

    def create_conversation(self, title: str = "New conversation") -> str:
        cid = new_id("conv")
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at)"
                " VALUES (?,?,?,?)", (cid, title, now, now))
        return cid

    def list_conversations(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m"
                "  WHERE m.conversation_id = c.id) AS message_count"
                " FROM conversations c ORDER BY c.updated_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,)).fetchone()
        return dict(row) if row else None

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ?"
                " WHERE id = ?", (title, time.time(), conversation_id))

    def set_summary(self, conversation_id: str, summary: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET summary = ?, updated_at = ?"
                " WHERE id = ?", (summary, time.time(), conversation_id))

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?",
                         (conversation_id,))

    # ---- messages ------------------------------------------------------

    def add_message(self, conversation_id: str, role: str, content: str,
                    reasoning: str = "", citations=None,
                    metadata=None) -> str:
        mid = new_id("msg")
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content,"
                " reasoning, citations, metadata, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (mid, conversation_id, role, content, reasoning,
                 json.dumps(citations or []), json.dumps(metadata or {}), now))
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id))
        return mid

    def get_messages(self, conversation_id: str,
                     limit: int | None = None) -> list[dict]:
        sql = ("SELECT * FROM messages WHERE conversation_id = ?"
               " ORDER BY created_at ASC")
        args: tuple = (conversation_id,)
        if limit is not None:
            sql += " LIMIT ?"
            args = (conversation_id, limit)
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["citations"] = json.loads(item["citations"])
            item["metadata"] = json.loads(item["metadata"])
            out.append(item)
        return out

    # ---- collections and documents -------------------------------------

    def create_collection(self, name: str) -> str:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT id FROM collections WHERE name = ?",
                               (name,)).fetchone()
            if row:
                return row["id"]
            cid = new_id("coll")
            conn.execute(
                "INSERT INTO collections (id, name, created_at) VALUES (?,?,?)",
                (cid, name, time.time()))
        return cid

    def list_collections(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM documents d"
                "  WHERE d.collection_id = c.id) AS document_count"
                " FROM collections c ORDER BY c.name").fetchall()
        return [dict(r) for r in rows]

    def add_document(self, collection_id: str, source_path: str,
                     filename: str, media_type: str, sha256: str,
                     page_count: int = 0, metadata=None) -> str | None:
        """Insert a document; returns None when the sha is already present."""
        did = new_id("doc")
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM documents WHERE collection_id = ?"
                " AND sha256 = ?", (collection_id, sha256)).fetchone()
            if existing:
                return None
            conn.execute(
                "INSERT INTO documents (id, collection_id, source_path,"
                " filename, media_type, sha256, page_count, metadata,"
                " created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (did, collection_id, source_path, filename, media_type,
                 sha256, page_count, json.dumps(metadata or {}), time.time()))
        return did

    def list_documents(self, collection_id: str | None = None) -> list[dict]:
        sql = ("SELECT d.*, (SELECT COUNT(*) FROM chunks c"
               " WHERE c.document_id = d.id) AS chunk_count FROM documents d")
        args: tuple = ()
        if collection_id:
            sql += " WHERE d.collection_id = ?"
            args = (collection_id,)
        sql += " ORDER BY d.created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, document_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def add_chunks(self, document_id: str, chunks: list[dict]) -> int:
        """Insert chunks. Each dict: ordinal, text, page, heading, embedding."""
        now = time.time()
        rows = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            rows.append((
                new_id("chunk"), document_id, chunk["ordinal"],
                chunk.get("page"), chunk.get("heading", ""), chunk["text"],
                chunk.get("token_count", 0),
                pack_vector(embedding) if embedding is not None else None,
                now))
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO chunks (id, document_id, ordinal, page, heading,"
                " text, token_count, embedding, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def iter_chunks(self, collection_id: str | None = None) -> list[dict]:
        """Load chunks with their documents for retrieval."""
        sql = ("SELECT ch.id, ch.document_id, ch.ordinal, ch.page,"
               " ch.heading, ch.text, ch.embedding, d.filename, d.source_path,"
               " d.collection_id FROM chunks ch"
               " JOIN documents d ON d.id = ch.document_id")
        args: tuple = ()
        if collection_id:
            sql += " WHERE d.collection_id = ?"
            args = (collection_id,)
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            blob = item.pop("embedding")
            item["embedding"] = unpack_vector(blob) if blob else None
            out.append(item)
        return out

    def get_chunk(self, chunk_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ch.*, d.filename, d.source_path FROM chunks ch"
                " JOIN documents d ON d.id = ch.document_id"
                " WHERE ch.id = ?", (chunk_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        blob = item.pop("embedding")
        item["embedding"] = unpack_vector(blob) if blob else None
        return item

    # ---- memory --------------------------------------------------------

    def propose_memory(self, kind: str, content: str,
                       provenance: str = "") -> str:
        """Record a candidate memory. Nothing becomes durable until approved."""
        mid = new_id("mem")
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO memories (id, kind, content, provenance, status,"
                " created_at, updated_at) VALUES (?,?,?,?,'proposed',?,?)",
                (mid, kind, content, provenance, now, now))
        return mid

    def list_memories(self, status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM memories"
        args: tuple = ()
        if status:
            sql += " WHERE status = ?"
            args = (status,)
        sql += " ORDER BY updated_at DESC"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, args).fetchall()]

    def set_memory_status(self, memory_id: str, status: str) -> None:
        if status not in ("proposed", "approved", "rejected"):
            raise ValueError(f"unknown memory status {status!r}")
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), memory_id))

    def update_memory(self, memory_id: str, content: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                (content, time.time(), memory_id))

    def delete_memory(self, memory_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    # ---- decisions and audit -------------------------------------------

    def record_decision(self, record: dict, state, criterion: str,
                        options: list, conversation_id: str | None = None):
        did = new_id("dec")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO decisions (id, conversation_id, row_id, state,"
                " criterion, options, record, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (did, conversation_id, record.get("id", ""),
                 json.dumps(state, ensure_ascii=False), criterion,
                 json.dumps(options, ensure_ascii=False),
                 json.dumps(record, ensure_ascii=False, default=str),
                 time.time()))
        return did

    def list_decisions(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["record"] = json.loads(item["record"])
            item["options"] = json.loads(item["options"])
            out.append(item)
        return out

    def record_audit(self, tool: str, arguments: dict, decision: str,
                     result_summary: str = "", error: str = "",
                     duration_s: float = 0.0,
                     conversation_id: str | None = None) -> str:
        aid = new_id("audit")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO audit (id, conversation_id, tool, arguments,"
                " decision, result_summary, error, duration_s, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (aid, conversation_id, tool,
                 json.dumps(arguments, ensure_ascii=False, default=str),
                 decision, result_summary[:2000], error[:2000], duration_s,
                 time.time()))
        return aid

    def list_audit(self, limit: int = 200) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["arguments"] = json.loads(item["arguments"])
            out.append(item)
        return out
