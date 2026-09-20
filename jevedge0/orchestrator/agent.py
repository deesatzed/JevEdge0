"""The agent loop: model proposes, gateway disposes.

One iteration is: render the conversation and tool descriptions, ask
Edge0 for a JSON envelope, validate it, execute at most one allowlisted
tool, feed the result back.  Repeat until the model answers or the step
budget runs out.

Malformed output is retried with the parse error fed back to the model,
never coerced into a best-guess call.  A tool that cannot be parsed
confidently is a tool that must not run.
"""

from __future__ import annotations

import json

from jevedge0.orchestrator.memory import MemoryManager
from jevedge0.tools.protocol import (ProtocolError, build_system_prompt,
                                     format_result, parse_envelope)
from jevedge0.tools.registry import PermissionDenied, ToolError

MAX_STEPS = 6
MAX_PARSE_RETRIES = 2


class PendingConfirmation(Exception):
    """Raised when a tool needs the user's explicit go-ahead."""

    def __init__(self, tool: str, arguments: dict, message: str):
        super().__init__(message)
        self.tool = tool
        self.arguments = arguments


class Agent:
    """Tool-using agent over an Edge0 client."""

    def __init__(self, client, registry, store=None, knowledge=None,
                 max_steps: int = MAX_STEPS):
        self.client = client
        self.registry = registry
        self.store = store
        self.knowledge = knowledge
        self.max_steps = max_steps
        self.memory = MemoryManager(store) if store is not None else None

    # ---- prompt assembly -------------------------------------------------

    def system_prompt(self) -> str:
        prompt = build_system_prompt(self.registry.specs())
        if self.memory is not None:
            block = self.memory.context_block()
            if block:
                prompt += "\n\n" + block
        return prompt

    def _history(self, conversation_id: str | None,
                 limit: int = 12) -> list[dict]:
        if not conversation_id or self.store is None:
            return []
        conversation = self.store.get_conversation(conversation_id)
        messages = self.store.get_messages(conversation_id)
        out = []
        if conversation and conversation.get("summary"):
            out.append({
                "role": "user",
                "content": ("Summary of earlier conversation:\n"
                            + conversation["summary"]),
            })
        for message in messages[-limit:]:
            if message["role"] in ("user", "assistant") and message["content"]:
                out.append({"role": message["role"],
                            "content": message["content"]})
        return out

    # ---- main loop -------------------------------------------------------

    def run(self, question: str, conversation_id: str | None = None,
            approved_tools: set[str] | None = None,
            on_event=None) -> dict:
        """Answer ``question``, using tools as needed."""
        approved = approved_tools or set()
        known = self.registry.available_names()
        transcript = [
            {"role": "system", "content": self.system_prompt()},
            *self._history(conversation_id),
            {"role": "user", "content": question},
        ]
        steps = []

        def emit(kind: str, **data):
            if on_event:
                on_event({"type": kind, **data})

        for step in range(self.max_steps):
            envelope, raw = self._ask(transcript, known, emit)
            if envelope is None:
                return self._finish(
                    question,
                    "I could not produce a valid response after several "
                    "attempts. The last reply was:\n\n" + raw.strip()[:800],
                    steps, conversation_id, failed=True)

            if envelope["action"] == "answer":
                emit("answer", content=envelope["content"])
                return self._finish(question, envelope["content"], steps,
                                    conversation_id,
                                    citations=envelope.get("citations", []))

            name = envelope["tool"]
            arguments = envelope["arguments"]
            emit("tool_call", tool=name, arguments=arguments)

            try:
                outcome = self.registry.invoke(
                    name, arguments, approved=name in approved,
                    conversation_id=conversation_id)
                payload = format_result(name, outcome["result"])
                steps.append({"tool": name, "arguments": arguments,
                              "status": "executed",
                              "result": outcome["result"]})
                emit("tool_result", tool=name, result=outcome["result"])
            except PermissionDenied as exc:
                if exc.needs_confirmation:
                    emit("confirmation_required", tool=name,
                         arguments=arguments)
                    raise PendingConfirmation(name, arguments, str(exc))
                payload = format_result(name, None, error=str(exc))
                steps.append({"tool": name, "arguments": arguments,
                              "status": "refused", "error": str(exc)})
                emit("tool_refused", tool=name, error=str(exc))
            except (ToolError, ProtocolError) as exc:
                payload = format_result(name, None, error=str(exc))
                steps.append({"tool": name, "arguments": arguments,
                              "status": "failed", "error": str(exc)})
                emit("tool_failed", tool=name, error=str(exc))

            transcript.append({"role": "assistant",
                               "content": json.dumps({
                                   "action": "tool", "tool": name,
                                   "arguments": arguments})})
            transcript.append({"role": "user",
                               "content": f"Tool result: {payload}"})

        # Budget exhausted: ask for a final answer from what was gathered.
        transcript.append({
            "role": "user",
            "content": ("You have used the available tool steps. Answer now "
                        "from what the tool results established, and say "
                        "plainly what remains unresolved. Reply with the "
                        'answer envelope: {"action": "answer", "content": '
                        '"..."}'),
        })
        raw = self.client.chat(transcript, max_tokens=1200)
        try:
            envelope = parse_envelope(raw, known)
            content = (envelope["content"] if envelope["action"] == "answer"
                       else raw.strip())
        except ProtocolError:
            content = raw.strip()
        return self._finish(question, content, steps, conversation_id,
                            budget_exhausted=True)

    def _ask(self, transcript: list[dict], known: set[str], emit):
        """Get one valid envelope, retrying on parse failure."""
        raw = ""
        for attempt in range(MAX_PARSE_RETRIES + 1):
            raw = self.client.chat(transcript, max_tokens=900)
            try:
                return parse_envelope(raw, known), raw
            except ProtocolError as exc:
                emit("parse_error", error=str(exc), attempt=attempt + 1)
                if attempt == MAX_PARSE_RETRIES:
                    break
                transcript.append({"role": "assistant", "content": raw})
                transcript.append({
                    "role": "user",
                    "content": (f"That was not valid: {exc}. Reply with one "
                                "JSON object only, no other text, using "
                                'either {"action":"tool","tool":"<name>",'
                                '"arguments":{...}} or {"action":"answer",'
                                '"content":"..."}.'),
                })
        return None, raw

    def _finish(self, question: str, answer: str, steps: list,
                conversation_id: str | None, citations=None,
                failed: bool = False,
                budget_exhausted: bool = False) -> dict:
        if conversation_id and self.store is not None:
            self.store.add_message(conversation_id, "user", question)
            self.store.add_message(
                conversation_id, "assistant", answer,
                citations=citations or [],
                metadata={"steps": len(steps),
                          "tools": [s["tool"] for s in steps]})
        return {
            "answer": answer,
            "steps": steps,
            "citations": citations or [],
            "failed": failed,
            "budget_exhausted": budget_exhausted,
            "conversation_id": conversation_id,
        }
