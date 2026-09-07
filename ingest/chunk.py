import hashlib
import uuid

from app.models import Chunk
from ingest.parse import PageBlock


def content_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split a single paragraph that on its own exceeds max_chars.

    Falls back to splitting on single newlines first (still respecting
    natural line boundaries), and if an individual line is still too long,
    splits on whitespace at roughly max_chars-sized boundaries so no chunk
    ever meaningfully exceeds max_chars regardless of input structure.
    """
    lines = [line for line in paragraph.split("\n") if line.strip()]
    if len(lines) <= 1:
        lines = [paragraph]

    out: list[str] = []
    buf = ""
    for line in lines:
        if len(line) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            words = line.split(" ")
            piece = ""
            for word in words:
                candidate = f"{piece} {word}" if piece else word
                if piece and len(candidate) > max_chars:
                    out.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                buf = piece
        elif buf and len(buf) + len(line) + 1 > max_chars:
            out.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        out.append(buf)
    return out


def _split_prose(text: str, max_chars: int) -> list[str]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []
    out: list[str] = []
    buf = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_oversized_paragraph(paragraph, max_chars))
        elif buf and len(buf) + len(paragraph) + 2 > max_chars:
            out.append(buf)
            buf = paragraph
        else:
            buf = f"{buf}\n\n{paragraph}" if buf else paragraph
    if buf:
        out.append(buf)
    return out


def blocks_to_chunks(doc_id: str, blocks: list[PageBlock], max_prose_chars: int = 1500) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in blocks:
        texts = [block.text] if block.kind == "table" else _split_prose(block.text, max_prose_chars)
        offset = block.char_start
        for text in texts:
            if not text.strip():
                continue
            chunks.append(Chunk(
                id=str(uuid.uuid4()), doc_id=doc_id, page_no=block.page_no, kind=block.kind,
                text=text, char_start=offset, content_hash=content_hash(text), score=0.0, extracted=False,
            ))
            offset += len(text)
    return chunks
