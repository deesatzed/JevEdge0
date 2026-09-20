"""Document extraction and structure-aware chunking.

Extraction keeps the provenance a citation needs: which file, which page,
which heading.  A chunk that cannot say where it came from cannot be
cited, and an assistant that cannot cite should abstain rather than
assert.

Chunking respects document structure first (headings, pages, rows) and
falls back to sentence-boundary packing inside a section.  Splitting on a
fixed character count would cut sentences in half and strand the subject
of a claim in a different chunk from its object.
"""

from __future__ import annotations

import csv
import hashlib
import html.parser
import io
import os
import re

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log"}
SUPPORTED = TEXT_SUFFIXES | {".pdf", ".docx", ".html", ".htm", ".csv", ".tsv"}


class Segment:
    """One extracted region of a document, with its provenance."""

    __slots__ = ("text", "page", "heading")

    def __init__(self, text: str, page: int | None = None, heading: str = ""):
        self.text = text
        self.page = page
        self.heading = heading

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Segment(page={self.page}, heading={self.heading!r})"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- extractors --------------------------------------------------------

def _extract_pdf(path: str) -> list[Segment]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    segments = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            segments.append(Segment(_clean(text), page=number))
    if not segments:
        raise ValueError(
            f"{os.path.basename(path)}: no extractable text — this looks "
            "like a scanned PDF; OCR it before ingesting")
    return segments


def _extract_docx(path: str) -> list[Segment]:
    import docx
    document = docx.Document(path)
    segments = []
    heading = ""
    buffer: list[str] = []

    def flush():
        if buffer:
            segments.append(Segment(_clean("\n".join(buffer)), heading=heading))
            buffer.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style.name.startswith("Heading"):
            flush()
            heading = text
            continue
        buffer.append(text)
    flush()

    for index, table in enumerate(document.tables, start=1):
        rows = ["\t".join(cell.text.strip() for cell in row.cells)
                for row in table.rows]
        body = "\n".join(r for r in rows if r.strip())
        if body:
            segments.append(Segment(body, heading=f"Table {index}"))
    return segments


class _HTMLText(html.parser.HTMLParser):
    """Collect visible text, tracking the most recent heading."""

    SKIP = {"script", "style", "noscript", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.segments: list[Segment] = []
        self._buffer: list[str] = []
        self._heading = ""
        self._skip_depth = 0
        self._in_heading = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._in_heading = True
        elif tag in ("p", "div", "li", "tr", "br", "section"):
            self._flush()

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif re.fullmatch(r"h[1-6]", tag):
            self._heading = " ".join(self._buffer).strip()
            self._buffer.clear()
            self._in_heading = False

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._buffer.append(text)

    def _flush(self):
        if self._in_heading:
            return
        text = " ".join(self._buffer).strip()
        self._buffer.clear()
        if text:
            self.segments.append(Segment(text, heading=self._heading))

    def close(self):
        super().close()
        self._flush()


def _extract_html(path: str) -> list[Segment]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        parser = _HTMLText()
        parser.feed(fh.read())
        parser.close()
    return parser.segments


def _extract_csv(path: str) -> list[Segment]:
    delimiter = "\t" if path.endswith(".tsv") else ","
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            pass
        rows = list(csv.reader(fh, delimiter=delimiter))
    if not rows:
        return []
    header = rows[0]
    segments = []
    # One segment per row, labelled by header, so a retrieved row still
    # says which column each value belongs to.
    for index, row in enumerate(rows[1:], start=1):
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v.strip()]
        if pairs:
            segments.append(Segment("; ".join(pairs),
                                    heading=f"row {index}"))
    return segments


def _extract_markdown(path: str) -> list[Segment]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    segments = []
    heading = ""
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            segments.append(Segment(body, heading=heading))

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if match:
            flush()
            heading = match.group(2).strip()
            continue
        buffer.append(line)
    flush()
    return segments


def _extract_text(path: str) -> list[Segment]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        body = _clean(fh.read())
    return [Segment(body)] if body.strip() else []


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract(path: str) -> tuple[list[Segment], str, int]:
    """Extract a document. Returns ``(segments, media_type, page_count)``."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in SUPPORTED:
        raise ValueError(
            f"unsupported file type {suffix!r}; supported: "
            f"{', '.join(sorted(SUPPORTED))}")
    if suffix == ".pdf":
        segments = _extract_pdf(path)
        media = "application/pdf"
    elif suffix == ".docx":
        segments = _extract_docx(path)
        media = ("application/vnd.openxmlformats-officedocument"
                 ".wordprocessingml.document")
    elif suffix in (".html", ".htm"):
        segments = _extract_html(path)
        media = "text/html"
    elif suffix in (".csv", ".tsv"):
        segments = _extract_csv(path)
        media = "text/csv"
    elif suffix in (".md", ".markdown"):
        segments = _extract_markdown(path)
        media = "text/markdown"
    else:
        segments = _extract_text(path)
        media = "text/plain"
    pages = {s.page for s in segments if s.page is not None}
    return segments, media, len(pages)


# ---- chunking ----------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_segments(segments: list[Segment], target_chars: int = 1200,
                   overlap_chars: int = 180,
                   min_chars: int = 80) -> list[dict]:
    """Pack segments into chunks that never span a page or heading change.

    Overlap carries the tail of one chunk into the next so a claim split
    across a boundary is still retrievable from at least one chunk whole.
    """
    chunks: list[dict] = []
    ordinal = 0

    def emit(text: str, page, heading: str):
        nonlocal ordinal
        body = text.strip()
        if len(body) < min_chars and chunks and chunks[-1]["page"] == page:
            # Too small to stand alone: append to the previous chunk
            # rather than creating a fragment nothing can cite usefully.
            chunks[-1]["text"] = f"{chunks[-1]['text']}\n{body}".strip()
            return
        if not body:
            return
        chunks.append({
            "ordinal": ordinal, "text": body, "page": page,
            "heading": heading, "token_count": max(1, len(body) // 4),
        })
        ordinal += 1

    for segment in segments:
        if len(segment.text) <= target_chars:
            emit(segment.text, segment.page, segment.heading)
            continue
        current = ""
        for sentence in split_sentences(segment.text):
            if len(sentence) > target_chars:
                # A single oversized sentence (dense tables, minified
                # text): hard-split it, since no boundary exists.
                if current:
                    emit(current, segment.page, segment.heading)
                    current = ""
                for start in range(0, len(sentence), target_chars):
                    emit(sentence[start:start + target_chars],
                         segment.page, segment.heading)
                continue
            if len(current) + len(sentence) + 1 > target_chars and current:
                emit(current, segment.page, segment.heading)
                tail = current[-overlap_chars:] if overlap_chars else ""
                # Resume at a sentence boundary inside the overlap.
                pieces = split_sentences(tail)
                current = (pieces[-1] + " " + sentence) if pieces else sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            emit(current, segment.page, segment.heading)
    return chunks


def ingest_file(path: str, target_chars: int = 1200) -> dict:
    """Extract and chunk one file (no embedding, no storage)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    segments, media_type, page_count = extract(path)
    chunks = chunk_segments(segments, target_chars=target_chars)
    if not chunks:
        raise ValueError(f"{os.path.basename(path)}: no text extracted")
    return {
        "source_path": os.path.abspath(path),
        "filename": os.path.basename(path),
        "media_type": media_type,
        "sha256": file_sha256(path),
        "page_count": page_count,
        "chunks": chunks,
    }
