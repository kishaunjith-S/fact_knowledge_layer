# app/models.py
from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class Document:
    id: str
    filename: str
    sha256: str
    page_count: int
    title: str | None
    publisher: str | None
    doc_date: date | None
    uploaded_at: datetime
    status: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    page_no: int
    kind: str
    text: str
    char_start: int
    content_hash: str
    score: float
    extracted: bool


@dataclass
class Claim:
    id: str
    doc_id: str
    chunk_id: str
    subject: str
    subject_key: str
    measure: str
    measure_key: str
    value_type: str
    value_num: float | None
    value_text: str | None
    unit_raw: str | None
    unit_dim: str | None
    unit_scale: float
    period_start: date | None
    period_end: date | None
    period_label: str | None
    basis: dict
    evidence_page: int
    evidence_quote: str
    evidence_char_start: int | None
    confidence: float
    extractor_version: str
    doc_vintage: date | None = None  # joined from documents.doc_date; not a claims column


@dataclass
class Relation:
    id: str
    claim_a_id: str
    claim_b_id: str
    rule_verdict: str
    final_verdict: str
    axis: str | None
    rule_fired: str
    delta_pct: float | None
    explanation: str | None = None
    llm_overrode: bool = False
    override_reason: str | None = None
