"""Tool registry, permission levels, and the audited execution gateway.

Permission levels (from the policy table in ``edg1.md``):

``automatic``   runs without asking — read-only and low consequence
``confirm``     requires explicit per-call approval from the user
``disabled``    registered but refused; enabling is a deliberate act

Every invocation is audited whether it ran, was refused, or failed.  An
audit trail that only records successes cannot answer the question people
actually ask after an incident, which is what the assistant *tried* to do.
"""

from __future__ import annotations

import fnmatch
import os
import time

from jevedge0.tools.protocol import ProtocolError, validate_arguments

AUTOMATIC = "automatic"
CONFIRM = "confirm"
DISABLED = "disabled"
LEVELS = (AUTOMATIC, CONFIRM, DISABLED)


class ToolError(RuntimeError):
    """Raised when a tool cannot complete its work."""


class PermissionDenied(RuntimeError):
    """Raised when policy refuses a call."""

    def __init__(self, message: str, needs_confirmation: bool = False):
        super().__init__(message)
        self.needs_confirmation = needs_confirmation


class Policy:
    """Filesystem and capability boundaries for the tool gateway.

    Path access is allowlist-only: a folder must be named before anything
    under it can be read.  Denying a blocklist would mean guessing every
    sensitive location in advance, and the first one missed is the one
    that matters.
    """

    def __init__(self, allowed_folders=None, workspace: str | None = None,
                 allow_network: bool = False,
                 denied_globs=("*.env", "*.pem", "*.key", "*id_rsa*",
                               "*credentials*", "*.sqlite-wal")):
        self.allowed_folders = [os.path.realpath(os.path.expanduser(p))
                                for p in (allowed_folders or [])]
        self.workspace = (os.path.realpath(os.path.expanduser(workspace))
                          if workspace else None)
        if self.workspace:
            os.makedirs(self.workspace, exist_ok=True)
            if self.workspace not in self.allowed_folders:
                self.allowed_folders.append(self.workspace)
        self.allow_network = allow_network
        self.denied_globs = list(denied_globs)

    def resolve_readable(self, path: str) -> str:
        """Resolve a path that a tool may read, or raise."""
        target = os.path.realpath(os.path.expanduser(path))
        if not self.allowed_folders:
            raise PermissionDenied(
                "no folders are approved for file access; add one in settings")
        for folder in self.allowed_folders:
            if target == folder or target.startswith(folder + os.sep):
                break
        else:
            raise PermissionDenied(
                f"path is outside every approved folder: {path}")
        name = os.path.basename(target)
        for pattern in self.denied_globs:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(target, pattern):
                raise PermissionDenied(
                    f"path matches a protected pattern ({pattern}): {name}")
        return target

    def resolve_writable(self, path: str) -> str:
        """Resolve a path a tool may write. Writes stay in the workspace."""
        if not self.workspace:
            raise PermissionDenied("no workspace is configured for writing")
        target = os.path.realpath(os.path.join(
            self.workspace, os.path.expanduser(path)))
        if not (target == self.workspace
                or target.startswith(self.workspace + os.sep)):
            raise PermissionDenied(
                "writes are confined to the workspace directory")
        return target


class Tool:
    """One registered capability."""

    def __init__(self, name: str, description: str, arguments: dict,
                 handler, permission: str = AUTOMATIC):
        if permission not in LEVELS:
            raise ValueError(f"unknown permission level {permission!r}")
        self.name = name
        self.description = description
        self.arguments = arguments
        self.handler = handler
        self.permission = permission

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
            "permission": self.permission,
        }


class ToolRegistry:
    """Registry + audited execution gateway."""

    def __init__(self, store=None, policy: Policy | None = None):
        self.store = store
        self.policy = policy or Policy()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def add(self, name: str, description: str, arguments: dict, handler,
            permission: str = AUTOMATIC) -> None:
        self.register(Tool(name, description, arguments, handler, permission))

    def set_permission(self, name: str, permission: str) -> None:
        if name not in self._tools:
            raise KeyError(name)
        if permission not in LEVELS:
            raise ValueError(f"unknown permission level {permission!r}")
        self._tools[name].permission = permission

    @property
    def names(self) -> set[str]:
        return set(self._tools)

    def specs(self, include_disabled: bool = False) -> list[dict]:
        return [t.spec() for t in self._tools.values()
                if include_disabled or t.permission != DISABLED]

    def available_names(self) -> set[str]:
        return {n for n, t in self._tools.items() if t.permission != DISABLED}

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ProtocolError(f"unknown tool {name!r}")
        return self._tools[name]

    # ---- execution ------------------------------------------------------

    def invoke(self, name: str, arguments: dict, approved: bool = False,
               conversation_id: str | None = None) -> dict:
        """Run one tool under policy, auditing the outcome either way."""
        started = time.perf_counter()
        tool = self.get(name)

        def audit(decision: str, summary: str = "", error: str = ""):
            if self.store is not None:
                self.store.record_audit(
                    tool=name, arguments=arguments, decision=decision,
                    result_summary=summary, error=error,
                    duration_s=time.perf_counter() - started,
                    conversation_id=conversation_id)

        if tool.permission == DISABLED:
            audit("refused", error="tool is disabled")
            raise PermissionDenied(f"tool {name!r} is disabled by policy")

        if tool.permission == CONFIRM and not approved:
            audit("awaiting_confirmation")
            raise PermissionDenied(
                f"tool {name!r} requires explicit confirmation",
                needs_confirmation=True)

        try:
            cleaned = validate_arguments(tool.spec(), arguments)
        except ProtocolError as exc:
            audit("rejected", error=str(exc))
            raise

        try:
            result = tool.handler(**cleaned)
        except (PermissionDenied, ToolError) as exc:
            audit("failed", error=str(exc))
            raise
        except Exception as exc:  # noqa: BLE001 - audit then surface
            audit("failed", error=f"{type(exc).__name__}: {exc}")
            raise ToolError(f"{name} failed: {exc}") from exc

        audit("executed", summary=str(result)[:2000])
        return {
            "tool": name,
            "result": result,
            "duration_s": time.perf_counter() - started,
        }
