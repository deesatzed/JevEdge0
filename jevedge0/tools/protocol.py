"""Structured tool-call protocol for a model without native tool calling.

Edge0 does not parse OpenAI ``tools`` or emit ``tool_calls``, so the
orchestrator asks for a strict JSON envelope instead and validates it
before anything runs.  The validation is the security boundary: a
malformed or unrecognized envelope is *rejected and retried*, never
guessed at and never executed.  "Parse loosely, execute anyway" is how a
model's formatting mistake turns into an unintended file write.

Envelope forms::

    {"action": "tool", "tool": "search_documents",
     "arguments": {"query": "..."}}

    {"action": "answer", "content": "...", "citations": ["#1"]}
"""

from __future__ import annotations

import json
import re

MAX_ARGUMENT_CHARS = 20000


class ProtocolError(ValueError):
    """Raised when a model response is not a valid tool envelope."""


def build_system_prompt(tool_specs: list[dict]) -> str:
    """Render the instruction block describing the available tools."""
    lines = [
        "You are a local assistant with access to tools.",
        "",
        "Reply with exactly one JSON object and nothing else. No prose "
        "before or after it, no markdown fences.",
        "",
        "To use a tool:",
        '  {"action": "tool", "tool": "<name>", "arguments": {...}}',
        "",
        "To answer the user:",
        '  {"action": "answer", "content": "<your answer>", '
        '"citations": ["#1"]}',
        "",
        "Rules:",
        "- Use only the tools listed below, with exactly these argument names.",
        "- One tool per reply. You will receive the result and may then call "
        "another tool or answer.",
        "- Cite evidence you were given as [#N] markers in your content.",
        "- If a tool fails or returns nothing useful, say so in your answer "
        "rather than inventing the result.",
        "- Never claim you performed an action that no tool result confirms.",
        "",
        "Available tools:",
    ]
    for spec in tool_specs:
        arguments = ", ".join(
            f"{name}: {meta.get('type', 'string')}"
            + ("" if meta.get("required", True) else " (optional)")
            for name, meta in spec.get("arguments", {}).items())
        lines.append(f"- {spec['name']}({arguments})")
        lines.append(f"    {spec['description']}")
        if spec.get("permission") != "automatic":
            lines.append(f"    permission: {spec['permission']}")
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    """Pull the JSON envelope out of a model response.

    Tolerates a markdown fence or surrounding whitespace — formatting
    noise around a well-formed object is a presentation slip, not an
    integrity problem.  It does not tolerate ambiguity: if no single
    balanced object can be isolated, this raises.
    """
    if not isinstance(text, str) or not text.strip():
        raise ProtocolError("empty model response")
    body = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.DOTALL)
    if fence:
        body = fence.group(1).strip()

    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed
        raise ProtocolError(f"expected a JSON object, got {type(parsed).__name__}")
    except json.JSONDecodeError:
        pass

    start = body.find("{")
    if start == -1:
        raise ProtocolError("no JSON object found in response")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = body[start:index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ProtocolError(
                        f"malformed JSON object: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise ProtocolError("expected a JSON object")
                return parsed
    raise ProtocolError("unterminated JSON object in response")


def parse_envelope(text: str, known_tools: set[str]) -> dict:
    """Validate a model response into a normalized envelope."""
    payload = extract_json(text)
    action = payload.get("action")
    if action not in ("tool", "answer"):
        raise ProtocolError(
            f"action must be 'tool' or 'answer', got {action!r}")

    if action == "answer":
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProtocolError("answer requires nonempty string 'content'")
        citations = payload.get("citations", [])
        if not isinstance(citations, list):
            raise ProtocolError("citations must be an array")
        return {"action": "answer", "content": content,
                "citations": [str(c) for c in citations]}

    name = payload.get("tool")
    if not isinstance(name, str) or not name:
        raise ProtocolError("tool call requires a string 'tool' name")
    if name not in known_tools:
        raise ProtocolError(
            f"unknown tool {name!r}; available: {sorted(known_tools)}")
    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ProtocolError("'arguments' must be an object")
    encoded = json.dumps(arguments, default=str)
    if len(encoded) > MAX_ARGUMENT_CHARS:
        raise ProtocolError(
            f"arguments too large ({len(encoded)} chars, "
            f"limit {MAX_ARGUMENT_CHARS})")
    return {"action": "tool", "tool": name, "arguments": arguments}


def validate_arguments(spec: dict, arguments: dict) -> dict:
    """Check arguments against a tool spec; returns the cleaned arguments."""
    declared = spec.get("arguments", {})
    unknown = set(arguments) - set(declared)
    if unknown:
        raise ProtocolError(
            f"{spec['name']}: unknown arguments {sorted(unknown)}; "
            f"expected {sorted(declared)}")
    cleaned = {}
    for name, meta in declared.items():
        if name not in arguments:
            if meta.get("required", True):
                raise ProtocolError(f"{spec['name']}: missing required "
                                    f"argument {name!r}")
            continue
        value = arguments[name]
        expected = meta.get("type", "string")
        checks = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "array": list, "object": dict,
        }
        wanted = checks.get(expected, object)
        if expected == "integer" and isinstance(value, bool):
            raise ProtocolError(f"{spec['name']}: {name} must be an integer")
        if not isinstance(value, wanted):
            raise ProtocolError(
                f"{spec['name']}: {name} must be {expected}, got "
                f"{type(value).__name__}")
        cleaned[name] = value
    return cleaned


def format_result(tool: str, result, error: str = "") -> str:
    """Render a tool result for the next model turn."""
    if error:
        return json.dumps({"tool": tool, "ok": False, "error": error},
                          ensure_ascii=False)
    return json.dumps({"tool": tool, "ok": True, "result": result},
                      ensure_ascii=False, default=str)
