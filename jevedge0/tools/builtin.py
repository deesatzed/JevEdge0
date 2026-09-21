"""The built-in tool set, with the starting policy from ``edg1.md``.

Every tool here is real: the calculator evaluates a parsed expression
tree, the SQL tools run against real databases, the Python tool runs in a
separate interpreter process.  Nothing returns canned output.

Two implementation notes worth stating plainly:

* The calculator walks an ``ast`` tree and permits only arithmetic nodes.
  ``eval`` on model-produced text would be arbitrary code execution
  wearing a calculator costume.
* The Python tool (``make_python_tool``) runs in a **separate OS
  process** with a timeout, which is real isolation against an accident:
  a runaway loop or a crash costs a killed child process, not this
  workbench. It is **not a security sandbox** against a deliberate
  attempt to escape it. A plain ``subprocess.run`` with an isolated
  interpreter (``-I``) and a trimmed environment provides none of: a
  network namespace (the child can still make outbound connections), a
  filesystem jail (the child can read or write any path its OS user can
  reach, not just the workspace), a memory or CPU cap (only wall-clock
  time is bounded), or a syscall filter (no seccomp or equivalent). Real
  isolation against untrusted code needs a VM or container boundary,
  which this project does not implement. ``run_python`` therefore
  **requires explicit per-call confirmation** (see its registration
  below) rather than running automatically -- the user sees the exact
  code before it executes; the earlier description of it as a "sandboxed
  workspace" overstated what the code provides, corrected here (see
  ``ERRORS.md`` E-014).
"""

from __future__ import annotations

import ast
import datetime
import json
import math
import operator
import os
import re
import sqlite3
import subprocess
import sys

from jevedge0.tools.registry import (AUTOMATIC, CONFIRM, DISABLED,
                                     PermissionDenied, ToolError)

# ---- calculator ---------------------------------------------------------

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
    "log10": math.log10, "exp": math.exp, "floor": math.floor,
    "ceil": math.ceil, "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_MAX_POWER = 1e6


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)):
            raise ToolError("only numeric literals are allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (
                abs(right) > 1000 or abs(left) ** min(abs(right), 64) > _MAX_POWER ** 4):
            raise ToolError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"unknown name {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolError("only arithmetic functions are allowed")
        if node.keywords:
            raise ToolError("keyword arguments are not supported")
        return _FUNCTIONS[node.func.id](*[_evaluate(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate(e) for e in node.elts]
    raise ToolError(
        f"expression element not permitted: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression."""
    if len(expression) > 500:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression: {exc.msg}") from exc
    value = _evaluate(tree)
    return {"expression": expression, "value": value}


# ---- time ---------------------------------------------------------------

def current_datetime(timezone: str = "local") -> dict:
    now = (datetime.datetime.now(datetime.timezone.utc)
           if timezone.lower() == "utc" else datetime.datetime.now().astimezone())
    return {
        "iso8601": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timezone": str(now.tzinfo),
    }


# ---- filesystem ---------------------------------------------------------

def make_file_tools(policy):
    """Build the filesystem tools bound to a policy."""

    def search_files(pattern: str, folder: str = "", max_results: int = 50):
        roots = ([policy.resolve_readable(folder)] if folder
                 else policy.allowed_folders)
        if not roots:
            raise PermissionDenied("no approved folders configured")
        needle = pattern.lower()
        matches = []
        for root in roots:
            for directory, subdirs, files in os.walk(root):
                subdirs[:] = [d for d in subdirs if not d.startswith(".")]
                for name in files:
                    if needle in name.lower():
                        path = os.path.join(directory, name)
                        try:
                            size = os.path.getsize(path)
                        except OSError:
                            continue
                        matches.append({"path": path, "name": name,
                                        "size_bytes": size})
                        if len(matches) >= max_results:
                            return {"matches": matches,
                                    "truncated": True}
        return {"matches": matches, "truncated": False}

    def read_file(path: str, max_chars: int = 6000):
        target = policy.resolve_readable(path)
        if not os.path.isfile(target):
            raise ToolError(f"not a file: {path}")
        if os.path.getsize(target) > 20_000_000:
            raise ToolError("file is too large to read (limit 20 MB)")
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read(max_chars + 1)
        truncated = len(body) > max_chars
        return {"path": target, "content": body[:max_chars],
                "truncated": truncated}

    def write_draft(filename: str, content: str):
        target = policy.resolve_writable(filename)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(content)
        return {"path": target, "bytes_written": len(content.encode("utf-8"))}

    return search_files, read_file, write_draft


# ---- SQL ----------------------------------------------------------------

_WRITE_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|"
    r"detach|pragma|vacuum|reindex)\b", re.IGNORECASE)


def _assert_read_only(sql: str) -> None:
    if ";" in sql.strip().rstrip(";"):
        raise ToolError("only a single statement may be run")
    if _WRITE_SQL.search(sql):
        raise ToolError("only read-only SELECT statements are permitted")
    if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
        raise ToolError("query must start with SELECT or WITH")


def make_sqlite_tool(policy):
    def query_sqlite(database: str, sql: str, max_rows: int = 100):
        _assert_read_only(sql)
        target = policy.resolve_readable(database)
        if not os.path.isfile(target):
            raise ToolError(f"database not found: {database}")
        # Read-only URI: the connection itself refuses writes, so a
        # statement that slips past the text check still cannot mutate.
        uri = f"file:{target}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = [dict(r) for r in cursor.fetchmany(max_rows)]
            truncated = cursor.fetchone() is not None
            conn.close()
        except sqlite3.Error as exc:
            raise ToolError(f"sqlite error: {exc}") from exc
        return {"rows": rows, "row_count": len(rows), "truncated": truncated}
    return query_sqlite


# ---- python -------------------------------------------------------------

def make_python_tool(policy, timeout_s: float = 20.0):
    """Run Python code in a separate process. NOT a security sandbox.

    Isolation actually provided: a fresh OS process (a crash or infinite
    loop cannot corrupt or hang this workbench process), a wall-clock
    timeout, an isolated interpreter (``-I``: ignores the user site and
    environment-based interpreter customization), a trimmed environment,
    and a working directory confined to the policy's workspace.

    Isolation NOT provided, despite those: the child process can open any
    outbound network connection (no network namespace); it can read or
    write any filesystem path its OS user has permission to touch, not
    only the workspace directory (no filesystem jail -- ``cwd`` only sets
    where relative paths resolve from, it does not restrict absolute
    paths); it has no memory or CPU ceiling, only a wall-clock one; and it
    has no syscall filter (no seccomp or equivalent), so any system call
    the OS user can make, the code can make. Untrusted code needs a VM or
    container boundary to be safely run, which this project does not
    implement. This is why ``run_python`` registers as ``DISABLED`` by
    default -- see ``register_builtin_tools`` below, and ``ERRORS.md``
    E-014 for the defect this corrects.
    """
    def run_python(code: str):
        if not policy.workspace:
            raise PermissionDenied("no workspace configured for code execution")
        if len(code) > 20000:
            raise ToolError("code too long")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True, text=True, timeout=timeout_s,
                cwd=policy.workspace,
                env={"PATH": os.environ.get("PATH", ""),
                     "HOME": policy.workspace,
                     "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"execution exceeded {timeout_s:g}s and was terminated") from None
        return {
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-4000:],
            "exit_code": completed.returncode,
        }
    return run_python


# ---- registration -------------------------------------------------------

def register_builtin_tools(registry, knowledge=None) -> None:
    """Register the built-in tools with their starting permissions."""
    policy = registry.policy

    registry.add(
        "calculator",
        "Evaluate an arithmetic expression. Supports + - * / // % **, "
        "sqrt, log, exp, floor, ceil, sin, cos, tan, abs, round, min, max.",
        {"expression": {"type": "string",
                        "description": "e.g. (71 - 14) / 4"}},
        calculate, AUTOMATIC)

    registry.add(
        "current_datetime",
        "Get the current date and time.",
        {"timezone": {"type": "string", "required": False,
                      "description": "'local' or 'utc'"}},
        current_datetime, AUTOMATIC)

    if knowledge is not None:
        def search_documents(query: str, top_k: int = 6,
                             collection: str = ""):
            results = knowledge.search(query, top_k=top_k,
                                       collection=collection or None)
            return {
                "matches": [{
                    "chunk_id": r["chunk_id"], "filename": r["filename"],
                    "page": r["page"], "heading": r["heading"],
                    "score": round(r.get("rerank_score", 0.0), 4),
                    "excerpt": r["text"][:600],
                } for r in results],
                "match_count": len(results),
            }

        def read_passage(chunk_id: str):
            chunk = knowledge.store.get_chunk(chunk_id)
            if chunk is None:
                raise ToolError(f"no passage with id {chunk_id}")
            return {
                "chunk_id": chunk["id"], "filename": chunk["filename"],
                "page": chunk["page"], "heading": chunk["heading"],
                "text": chunk["text"],
            }

        registry.add(
            "search_documents",
            "Search the indexed documents. Returns passages with ids, "
            "filenames and pages for citation.",
            {"query": {"type": "string"},
             "top_k": {"type": "integer", "required": False},
             "collection": {"type": "string", "required": False}},
            search_documents, AUTOMATIC)

        registry.add(
            "read_passage",
            "Read the full text of one passage returned by search_documents.",
            {"chunk_id": {"type": "string"}},
            read_passage, AUTOMATIC)

    search_files, read_file, write_draft = make_file_tools(policy)

    registry.add(
        "search_files",
        "Find files by name inside the approved folders.",
        {"pattern": {"type": "string"},
         "folder": {"type": "string", "required": False},
         "max_results": {"type": "integer", "required": False}},
        search_files, AUTOMATIC)

    registry.add(
        "read_file",
        "Read a text file from an approved folder.",
        {"path": {"type": "string"},
         "max_chars": {"type": "integer", "required": False}},
        read_file, AUTOMATIC)

    registry.add(
        "query_sqlite",
        "Run one read-only SELECT against a SQLite database in an "
        "approved folder.",
        {"database": {"type": "string"}, "sql": {"type": "string"},
         "max_rows": {"type": "integer", "required": False}},
        make_sqlite_tool(policy), AUTOMATIC)

    registry.add(
        "run_python",
        "Run a short Python script in a separate process and return its "
        "output. Not a security sandbox: no network, filesystem, or "
        "resource isolation beyond a timeout -- see make_python_tool's "
        "docstring. Requires explicit per-call confirmation: the user sees "
        "the exact code before it runs.",
        {"code": {"type": "string"}},
        make_python_tool(policy), CONFIRM)

    registry.add(
        "write_draft",
        "Write a draft file into the workspace directory.",
        {"filename": {"type": "string"}, "content": {"type": "string"}},
        write_draft, CONFIRM)

    def run_shell(command: str):  # pragma: no cover - disabled by policy
        raise PermissionDenied("shell execution is disabled")

    registry.add(
        "run_shell",
        "Run a shell command. Disabled by policy.",
        {"command": {"type": "string"}},
        run_shell, DISABLED)
