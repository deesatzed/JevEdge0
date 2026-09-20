"""Stdlib HTTP transport for the JevEdge0 workbench.

Edge0 keeps Flask optional behind a ``http.server`` fallback; JevEdge0
uses the stdlib path unconditionally so the workbench has no web
dependency at all and runs offline.

The server composes route tables: Edge0's own OpenAI-compatible handlers,
the decision endpoints, and (when the workbench is assembled) the chat,
knowledge, memory and audit routes.  Routes are plain
``"METHOD /path"`` keys mapping to callables, matching Edge0's
``build_app_handlers`` convention.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class Router:
    """Exact and pattern route table.

    Patterns use ``{name}`` segments (``GET /v1/chats/{chat_id}``); the
    captured values arrive as keyword arguments.
    """

    def __init__(self):
        self.exact: dict[str, callable] = {}
        self.patterns: list[tuple[str, re.Pattern, callable]] = []

    def add(self, key: str, handler) -> None:
        if "{" in key:
            method, path = key.split(" ", 1)
            regex = re.compile("^" + re.sub(
                r"\{(\w+)\}", r"(?P<\1>[^/]+)", path) + "$")
            self.patterns.append((method, regex, handler))
        else:
            self.exact[key] = handler

    def update(self, table: dict) -> None:
        for key, handler in table.items():
            self.add(key, handler)

    def resolve(self, method: str, path: str):
        handler = self.exact.get(f"{method} {path}")
        if handler is not None:
            return handler, {}
        for pattern_method, regex, pattern_handler in self.patterns:
            if pattern_method != method:
                continue
            match = regex.match(path)
            if match:
                return pattern_handler, match.groupdict()
        return None, {}


class StreamingResponse:
    """Marker for handlers that yield SSE byte chunks."""

    def __init__(self, events, content_type="text/event-stream; charset=utf-8"):
        self.events = events
        self.content_type = content_type


class FileResponse:
    """Marker for handlers that return static bytes."""

    def __init__(self, body: bytes, content_type: str):
        self.body = body
        self.content_type = content_type


class ErrorResponse:
    """Marker for an explicit HTTP status."""

    def __init__(self, status: int, message: str, kind="invalid_request"):
        self.status = status
        self.payload = {"error": {"message": message, "type": kind}}


class _Handler(BaseHTTPRequestHandler):
    router: Router = None
    protocol_version = "HTTP/1.1"
    quiet = True

    # ---- transport helpers -------------------------------------------

    def _send_json(self, status: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self, response: StreamingResponse) -> None:
        self.send_response(200)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()
        self.close_connection = True
        try:
            for chunk in response.events:
                if not chunk:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                self.wfile.write(f"{len(chunk):X}".encode("ascii")
                                 + b"\r\n" + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _cors(self) -> None:
        # The workbench binds to loopback; these headers only make the
        # bundled UI usable when opened from a file:// page.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, PUT, DELETE, OPTIONS")

    # ---- dispatch ------------------------------------------------------

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        handler, params = self.router.resolve(method, parsed.path)
        if handler is None:
            self._send_json(404, {"error": {
                "message": f"no route {method} {parsed.path}",
                "type": "not_found"}})
            return

        kwargs = dict(params)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if query:
            kwargs["query"] = query

        payload = None
        if method in ("POST", "PUT", "DELETE"):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if raw:
                ctype = (self.headers.get("Content-Type") or "")
                if "application/json" in ctype or raw[:1] in (b"{", b"["):
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        self._send_json(400, {"error": {
                            "message": f"invalid JSON body: {exc}",
                            "type": "invalid_request"}})
                        return
                else:
                    payload = raw
            else:
                payload = {}

        try:
            result = (handler(payload, **kwargs) if payload is not None
                      else handler(**kwargs))
        except Exception as exc:  # noqa: BLE001 - surface, never swallow
            traceback.print_exc()
            self._send_json(500, {"error": {"message": str(exc),
                                            "type": "server_error"}})
            return

        if isinstance(result, ErrorResponse):
            self._send_json(result.status, result.payload)
        elif isinstance(result, StreamingResponse):
            self._send_stream(result)
        elif isinstance(result, FileResponse):
            self._send_bytes(200, result.body, result.content_type)
        elif isinstance(result, tuple) and len(result) == 2:
            body, status = result
            self._send_json(status, body)
        else:
            status = 400 if isinstance(result, dict) and "error" in result else 200
            self._send_json(status, result)

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def do_PUT(self):  # noqa: N802
        self._dispatch("PUT")

    def do_DELETE(self):  # noqa: N802
        self._dispatch("DELETE")

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)


def static_handler(path: str):
    """Serve one file from disk, read per request so edits show up."""
    content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    if content_type.startswith("text/") or content_type.endswith("javascript"):
        content_type += "; charset=utf-8"

    def handler(**_):
        with open(path, "rb") as fh:
            return FileResponse(fh.read(), content_type)
    return handler


def serve(router: Router, host: str = "127.0.0.1", port: int = 8090,
          quiet: bool = True, background: bool = False):
    """Run the router. Returns the httpd when ``background`` is set."""
    handler_cls = type("JevEdge0Handler", (_Handler,),
                       {"router": router, "quiet": quiet})
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    httpd.daemon_threads = True
    if background:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd
    httpd.serve_forever()
    return httpd
