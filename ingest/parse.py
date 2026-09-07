import re
from dataclasses import dataclass
from datetime import date

import pymupdf

_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")


@dataclass(frozen=True)
class PageBlock:
    page_no: int
    kind: str
    text: str
    char_start: int


@dataclass(frozen=True)
class ParsedDocument:
    page_count: int
    doc_date: date | None
    blocks: list[PageBlock]


def _parse_pdf_date(raw: str | None) -> date | None:
    if not raw:
        return None
    match = _PDF_DATE_RE.match(raw)
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _table_rows_to_texts(header: list[str], rows: list[list[str]]) -> list[str]:
    texts = []
    for row in rows:
        cells = [str(c).strip() if c else "" for c in row]
        pairs = [f"{h}: {v}" for h, v in zip(header, cells) if v]
        if pairs:
            texts.append(" | ".join(pairs))
    return texts


def _blocks_excluding_tables(blocks, table_rects) -> list[str]:
    """Given page.get_text("blocks") tuples and a list of pymupdf.Rect table
    bounding boxes, return the text of every block that does NOT intersect
    any table rect, in reading order, with trailing whitespace stripped.

    This is what keeps prose from re-stating (in flattened, linearized form)
    the same numbers already extracted structurally as table rows.
    """
    kept: list[str] = []
    for block in blocks:
        block_rect = pymupdf.Rect(block[0], block[1], block[2], block[3])
        text = block[4]
        if any(block_rect.intersects(table_rect) for table_rect in table_rects):
            continue
        stripped = text.strip()
        if stripped:
            kept.append(stripped)
    return kept


def parse_pdf(path: str) -> ParsedDocument:
    doc = pymupdf.open(path)
    metadata = doc.metadata or {}
    doc_date = _parse_pdf_date(metadata.get("creationDate")) or _parse_pdf_date(metadata.get("modDate"))

    blocks: list[PageBlock] = []
    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        try:
            tables = page.find_tables().tables
        except Exception:
            tables = []

        char_start = 0
        table_rects = []
        for table in tables:
            rows = table.extract()
            if not rows or len(rows) < 2:
                continue
            header = [str(c).strip() if c else "" for c in rows[0]]
            row_texts = _table_rows_to_texts(header, rows[1:])
            if not row_texts:
                # This table produced no usable rows (e.g. every data cell was
                # empty) -- its bbox must NOT be excluded from prose, or the
                # text underneath it is dropped from the pipeline entirely
                # (emitted as neither a table block nor a prose block).
                continue
            table_rects.append(pymupdf.Rect(table.bbox))
            for row_text in row_texts:
                blocks.append(PageBlock(page_no=page_no, kind="table", text=row_text, char_start=char_start))
                char_start += len(row_text)

        prose_texts = _blocks_excluding_tables(page.get_text("blocks"), table_rects)
        prose_text = "\n\n".join(prose_texts)
        if prose_text.strip():
            blocks.append(PageBlock(page_no=page_no, kind="prose", text=prose_text, char_start=0))

    page_count = len(doc)
    doc.close()
    return ParsedDocument(page_count=page_count, doc_date=doc_date, blocks=blocks)
