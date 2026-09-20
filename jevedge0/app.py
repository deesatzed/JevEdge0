"""JevEdge0 workbench: assembles every layer over one Edge0 engine."""

from __future__ import annotations

import json
import os
import queue
import threading
import time

from jevedge0.comparator.dual import Comparator
from jevedge0.decision.scorer import Edge0DecisionScorer
from jevedge0.orchestrator.agent import Agent, PendingConfirmation
from jevedge0.orchestrator.engine_client import InProcessClient
from jevedge0.orchestrator.memory import MemoryManager
from jevedge0.rag.knowledge import KnowledgeService
from jevedge0.server.decisions import DecisionService
from jevedge0.server.http import (ErrorResponse, Router, StreamingResponse,
                                  static_handler)
from jevedge0.store.db import Store
from jevedge0.tools.builtin import register_builtin_tools
from jevedge0.tools.registry import Policy, ToolRegistry

DEFAULT_HOME = os.path.expanduser("~/.jevedge0")


class Workbench:
    """Every service, over one loaded engine."""

    def __init__(self, queue_server, home: str = DEFAULT_HOME,
                 allowed_folders=None, embedder=None,
                 load_embedder: bool = True):
        self.home = os.path.expanduser(home)
        os.makedirs(self.home, exist_ok=True)
        self.workspace = os.path.join(self.home, "workspace")
        os.makedirs(self.workspace, exist_ok=True)

        self.queue_server = queue_server
        self.store = Store(os.path.join(self.home, "jevedge0.db"))
        self.client = InProcessClient(queue_server)
        # DecisionService promotes the server's engine lock to an RLock;
        # hold that same object so a request can own the engine across a
        # decision *and* the generation it triggers (see engine_lock).
        self.decisions = DecisionService(queue_server)
        self.lock = self.decisions.lock
        self.scorer: Edge0DecisionScorer = self.decisions.scorer
        self.comparator = Comparator(self.scorer, self._chat)
        self.memory = MemoryManager(self.store)

        self.knowledge = None
        self.embedder_error = None
        if load_embedder:
            try:
                self.knowledge = KnowledgeService(self.store, embedder)
            except Exception as exc:  # noqa: BLE001 - surfaced in /health
                self.embedder_error = str(exc)

        self.policy = Policy(
            allowed_folders=allowed_folders or [self.workspace],
            workspace=self.workspace)
        self.registry = ToolRegistry(self.store, self.policy)
        register_builtin_tools(self.registry, self.knowledge)
        self.agent = Agent(self.client, self.registry, self.store,
                           self.knowledge)

    # ---- helpers ---------------------------------------------------------

    def _chat(self, messages, **kw) -> str:
        return self.client.chat(messages, **kw)

    @property
    def model_name(self) -> str:
        return self.queue_server.model_name

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": self.model_name,
            "home": self.home,
            "embeddings": {
                "available": self.knowledge is not None,
                "model": (self.knowledge.embedder.name
                          if self.knowledge else None),
                "error": self.embedder_error,
            },
            "tools": [t["name"] for t in self.registry.specs()],
            "collections": self.store.list_collections(),
        }


# ---- routes ---------------------------------------------------------------

def build_router(workbench: Workbench) -> Router:
    router = Router()
    store = workbench.store

    # --- static UI ---
    ui_path = os.path.join(os.path.dirname(__file__), "web", "ui.html")
    router.add("GET /", static_handler(ui_path))
    router.add("GET /index.html", static_handler(ui_path))

    # --- health / model ---
    router.add("GET /healthz", lambda **_: workbench.health())
    router.add("GET /v1/models", lambda **_: {
        "object": "list",
        "data": [{"id": workbench.model_name, "object": "model",
                  "owned_by": "jevedge0"}],
    })

    # --- chat (Edge0 passthrough, OpenAI-compatible) ---
    def chat_completions(payload, **_):
        from edge0.server.app import _chat_once, _chat_stream
        if payload.get("stream"):
            return StreamingResponse(
                _chat_stream(workbench.queue_server, payload))
        return _chat_once(workbench.queue_server, payload)

    router.add("POST /v1/chat/completions", chat_completions)

    # --- decisions ---
    from jevedge0.server.decisions import build_decision_handlers
    router.update(build_decision_handlers(workbench.queue_server,
                                          workbench.decisions))

    def compare(payload, **_):
        row = workbench.decisions._row_from_payload(payload)
        with workbench.lock:
            result = workbench.comparator.run(row)
        store.record_decision(result["decision"], row["state"],
                              row["question"], row["options"])
        decision = dict(result["decision"])
        decision["probabilities"] = dict(zip(
            decision["option_ids"], decision["probabilities"]))
        decision.pop("option_logits", None)
        decision.get("stability", {}).pop("trial_records", None)
        result["decision"] = decision
        return result

    router.add("POST /v1/compare", compare)

    # --- conversations ---
    router.add("GET /v1/conversations",
               lambda **_: {"conversations": store.list_conversations()})

    def create_conversation(payload, **_):
        title = (payload or {}).get("title") or "New conversation"
        return {"id": store.create_conversation(title), "title": title}

    router.add("POST /v1/conversations", create_conversation)
    router.add("GET /v1/conversations/{conversation_id}",
               lambda conversation_id, **_: {
                   "conversation": store.get_conversation(conversation_id),
                   "messages": store.get_messages(conversation_id)})

    def delete_conversation(payload=None, conversation_id=None, **_):
        store.delete_conversation(conversation_id)
        return {"deleted": conversation_id}

    router.add("DELETE /v1/conversations/{conversation_id}",
               delete_conversation)

    # --- agent ---
    def agent_ask(payload, **_):
        question = (payload or {}).get("question", "").strip()
        if not question:
            return ErrorResponse(400, "missing 'question'")
        conversation_id = payload.get("conversation_id")
        if not conversation_id:
            conversation_id = store.create_conversation(question[:60])
        approved = set(payload.get("approved_tools") or [])
        try:
            result = workbench.agent.run(
                question, conversation_id=conversation_id,
                approved_tools=approved)
        except PendingConfirmation as exc:
            return {"needs_confirmation": True, "tool": exc.tool,
                    "arguments": exc.arguments, "message": str(exc),
                    "conversation_id": conversation_id}
        result["conversation_id"] = conversation_id
        return result

    router.add("POST /v1/agent", agent_ask)

    def agent_stream(payload, **_):
        question = (payload or {}).get("question", "").strip()
        if not question:
            return ErrorResponse(400, "missing 'question'")
        conversation_id = (payload.get("conversation_id")
                           or store.create_conversation(question[:60]))
        approved = set(payload.get("approved_tools") or [])
        events: queue.Queue = queue.Queue()
        done = object()

        def emit(event):
            events.put(f"data: {json.dumps(event, default=str)}\n\n")

        def work():
            try:
                emit({"type": "start", "conversation_id": conversation_id})
                result = workbench.agent.run(
                    question, conversation_id=conversation_id,
                    approved_tools=approved, on_event=emit)
                emit({"type": "done", **result})
            except PendingConfirmation as exc:
                emit({"type": "confirmation_required", "tool": exc.tool,
                      "arguments": exc.arguments, "message": str(exc),
                      "conversation_id": conversation_id})
            except Exception as exc:  # noqa: BLE001 - report to the client
                emit({"type": "error", "message": str(exc)})
            finally:
                events.put("data: [DONE]\n\n")
                events.put(done)

        threading.Thread(target=work, daemon=True).start()

        def stream():
            while True:
                item = events.get()
                if item is done:
                    return
                yield item

        return StreamingResponse(stream())

    router.add("POST /v1/agent/stream", agent_stream)

    # --- knowledge ---
    def require_knowledge():
        return ErrorResponse(
            503, f"embeddings unavailable: {workbench.embedder_error}")

    def ingest(payload, **_):
        if workbench.knowledge is None:
            return require_knowledge()
        path = (payload or {}).get("path", "").strip()
        if not path:
            return ErrorResponse(400, "missing 'path'")
        path = os.path.expanduser(path)
        collection = payload.get("collection") or "default"
        try:
            if os.path.isdir(path):
                return workbench.knowledge.ingest_folder(path, collection)
            return workbench.knowledge.ingest(path, collection)
        except (FileNotFoundError, ValueError, NotADirectoryError) as exc:
            return ErrorResponse(400, str(exc))

    router.add("POST /v1/knowledge/ingest", ingest)
    router.add("GET /v1/knowledge/documents", lambda **kw: {
        "documents": store.list_documents(
            (kw.get("query") or {}).get("collection_id"))})
    router.add("GET /v1/knowledge/collections",
               lambda **_: {"collections": store.list_collections()})

    def delete_document(payload=None, document_id=None, **_):
        store.delete_document(document_id)
        if workbench.knowledge:
            workbench.knowledge._retrievers.clear()
        return {"deleted": document_id}

    router.add("DELETE /v1/knowledge/documents/{document_id}",
               delete_document)

    def search(payload, **_):
        if workbench.knowledge is None:
            return require_knowledge()
        query = (payload or {}).get("query", "").strip()
        if not query:
            return ErrorResponse(400, "missing 'query'")
        results = workbench.knowledge.search(
            query, top_k=int(payload.get("top_k", 8)),
            collection=payload.get("collection"))
        return {"results": results, "count": len(results)}

    router.add("POST /v1/knowledge/search", search)

    def ask_documents(payload, **_):
        if workbench.knowledge is None:
            return require_knowledge()
        question = (payload or {}).get("question", "").strip()
        if not question:
            return ErrorResponse(400, "missing 'question'")
        result = workbench.knowledge.answer(
            question, workbench._chat,
            top_k=int(payload.get("top_k", 6)),
            collection=payload.get("collection"))
        conversation_id = payload.get("conversation_id")
        if conversation_id:
            store.add_message(conversation_id, "user", question)
            store.add_message(conversation_id, "assistant", result["answer"],
                              citations=result.get("citations", []))
        return result

    router.add("POST /v1/knowledge/ask", ask_documents)

    # --- memory ---
    router.add("GET /v1/memory", lambda **kw: {
        "memories": store.list_memories((kw.get("query") or {}).get("status"))})

    def propose_memory(payload, **_):
        conversation_id = (payload or {}).get("conversation_id")
        if not conversation_id:
            return ErrorResponse(400, "missing 'conversation_id'")
        return {"proposed": workbench.memory.propose_from_conversation(
            conversation_id, workbench._chat)}

    router.add("POST /v1/memory/propose", propose_memory)

    def update_memory(payload, memory_id=None, **_):
        payload = payload or {}
        if "content" in payload:
            store.update_memory(memory_id, payload["content"])
        if "status" in payload:
            store.set_memory_status(memory_id, payload["status"])
        return {"updated": memory_id}

    router.add("PUT /v1/memory/{memory_id}", update_memory)

    def delete_memory(payload=None, memory_id=None, **_):
        store.delete_memory(memory_id)
        return {"deleted": memory_id}

    router.add("DELETE /v1/memory/{memory_id}", delete_memory)

    # --- tools and audit ---
    router.add("GET /v1/tools", lambda **_: {
        "tools": workbench.registry.specs(include_disabled=True)})

    def set_tool_permission(payload, name=None, **_):
        permission = (payload or {}).get("permission")
        try:
            workbench.registry.set_permission(name, permission)
        except (KeyError, ValueError) as exc:
            return ErrorResponse(400, str(exc))
        return {"tool": name, "permission": permission}

    router.add("PUT /v1/tools/{name}", set_tool_permission)
    router.add("GET /v1/audit", lambda **kw: {
        "audit": store.list_audit(int((kw.get("query") or {}).get("limit", 200)))})
    router.add("GET /v1/decisions/history",
               lambda **_: {"decisions": store.list_decisions()})

    return router
