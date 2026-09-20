"""Three kinds of memory, kept deliberately distinct.

``edg1.md`` is right that "memory" hides three different things:

* **Conversation history** — the messages of the current chat, complete.
* **Conversation summary** — older turns compressed to bound the prompt.
* **Durable memory** — facts that outlive the conversation.

Only the third is dangerous, because it is the one that silently follows
a person across every future conversation.  So durable memory here is
*proposed*, never auto-committed: the model can suggest, the user
approves, and everything approved is visible, editable and removable.
An assistant that decides on its own what to remember about someone has
taken a decision that was not its to take.
"""

from __future__ import annotations

import re

SUMMARY_SYSTEM = (
    "Summarize this conversation so it can be continued later.\n"
    "Keep decisions made, facts established, constraints stated, and open "
    "questions. Drop pleasantries and repetition. Write compact prose, no "
    "preamble. Do not invent anything that was not said."
)

PROPOSAL_SYSTEM = (
    "Identify durable facts worth remembering about the user or their "
    "ongoing work from this conversation.\n"
    "\n"
    "Qualifying: stable preferences, project facts, constraints, decisions "
    "with lasting effect.\n"
    "Not qualifying: one-off requests, transient state, anything already "
    "obvious from the current message, anything the user did not actually "
    "say.\n"
    "\n"
    "List at most three, one per line, each starting with '- '. Write "
    "nothing else. If nothing qualifies, reply exactly: NONE"
)


class MemoryManager:
    """Summarization and the propose/approve durable-memory workflow."""

    def __init__(self, store, history_limit: int = 20):
        self.store = store
        self.history_limit = history_limit

    # ---- durable memory --------------------------------------------------

    def approved(self) -> list[dict]:
        return self.store.list_memories(status="approved")

    def proposed(self) -> list[dict]:
        return self.store.list_memories(status="proposed")

    def context_block(self, limit: int = 20) -> str:
        """Render approved memories for the system prompt."""
        memories = self.approved()[:limit]
        if not memories:
            return ""
        lines = ["Durable memory (facts the user approved for you to keep):"]
        lines += [f"- {m['content']}" for m in memories]
        lines.append("Use these when relevant. If one contradicts what the "
                     "user says now, the user is right; say so.")
        return "\n".join(lines)

    def propose_from_conversation(self, conversation_id: str,
                                  chat_fn) -> list[dict]:
        """Ask the model for memory candidates. Nothing is stored approved."""
        messages = self.store.get_messages(conversation_id)
        if not messages:
            return []
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages[-30:])
        reply = chat_fn([
            {"role": "system", "content": PROPOSAL_SYSTEM},
            {"role": "user", "content": transcript},
        ])
        if reply.strip().upper().startswith("NONE"):
            return []

        existing = {m["content"].strip().lower()
                    for m in self.store.list_memories()}
        created = []
        for line in reply.splitlines():
            text = re.sub(r"^\s*[-*•]\s*", "", line).strip()
            if len(text) < 8 or text.upper() == "NONE":
                continue
            if text.strip().lower() in existing:
                continue
            memory_id = self.store.propose_memory(
                kind="fact", content=text,
                provenance=f"conversation {conversation_id}")
            created.append({"id": memory_id, "content": text,
                            "status": "proposed"})
            existing.add(text.strip().lower())
            if len(created) >= 3:
                break
        return created

    def approve(self, memory_id: str) -> None:
        self.store.set_memory_status(memory_id, "approved")

    def reject(self, memory_id: str) -> None:
        self.store.set_memory_status(memory_id, "rejected")

    def forget(self, memory_id: str) -> None:
        self.store.delete_memory(memory_id)

    # ---- summarization ---------------------------------------------------

    def maybe_summarize(self, conversation_id: str, chat_fn,
                        threshold: int | None = None) -> str | None:
        """Summarize older turns once the history outgrows the threshold."""
        threshold = threshold or self.history_limit
        messages = self.store.get_messages(conversation_id)
        if len(messages) < threshold:
            return None
        older = messages[:-8]
        if not older:
            return None
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in older)
        summary = chat_fn([
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": transcript},
        ]).strip()
        if summary:
            self.store.set_summary(conversation_id, summary)
        return summary or None
