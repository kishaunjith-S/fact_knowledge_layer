# app/db.py
import json
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path

from app.models import Claim, Chunk, Document, Relation

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    page_count INTEGER NOT NULL DEFAULT 0,
    title TEXT,
    publisher TEXT,
    doc_date TEXT,
    uploaded_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    page_no INTEGER NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    char_start INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    extracted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    chunk_id TEXT NOT NULL REFERENCES chunks(id),
    subject TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    measure TEXT NOT NULL,
    measure_key TEXT NOT NULL,
    value_type TEXT NOT NULL,
    value_num REAL,
    value_text TEXT,
    unit_raw TEXT,
    unit_dim TEXT,
    unit_scale REAL NOT NULL DEFAULT 1.0,
    period_start TEXT,
    period_end TEXT,
    period_label TEXT,
    basis TEXT NOT NULL DEFAULT '{}',
    evidence_page INTEGER NOT NULL,
    evidence_quote TEXT NOT NULL,
    evidence_char_start INTEGER,
    confidence REAL NOT NULL DEFAULT 0.5,
    extractor_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_group ON claims(subject_key, measure_key);
CREATE INDEX IF NOT EXISTS idx_claims_doc_id ON claims(doc_id);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    claim_a_id TEXT NOT NULL REFERENCES claims(id),
    claim_b_id TEXT NOT NULL REFERENCES claims(id),
    rule_verdict TEXT NOT NULL,
    final_verdict TEXT NOT NULL,
    axis TEXT,
    rule_fired TEXT NOT NULL,
    delta_pct REAL,
    explanation TEXT,
    llm_overrode INTEGER NOT NULL DEFAULT 0,
    override_reason TEXT,
    UNIQUE(claim_a_id, claim_b_id)
);
CREATE INDEX IF NOT EXISTS idx_relations_verdict ON relations(final_verdict);

CREATE TABLE IF NOT EXISTS measure_registry (
    measure_key TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    unit_dim TEXT,
    seen_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS failures (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    chunk_id TEXT REFERENCES chunks(id),
    kind TEXT NOT NULL DEFAULT 'extraction',
    reason TEXT NOT NULL,
    raw_payload TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_failures_doc_id ON failures(doc_id);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# ---- documents ----

def insert_document(conn, doc_id, filename, sha256, page_count, uploaded_at, status="pending"):
    conn.execute(
        "INSERT INTO documents (id, filename, sha256, page_count, uploaded_at, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (doc_id, filename, sha256, page_count, uploaded_at, status),
    )
    conn.commit()


def set_document_metadata(conn, doc_id, page_count, doc_date):
    conn.execute(
        "UPDATE documents SET page_count = ?, doc_date = ? WHERE id = ?",
        (page_count, doc_date.isoformat() if doc_date else None, doc_id),
    )
    conn.commit()


def update_document_status(conn, doc_id, status):
    conn.execute("UPDATE documents SET status = ? WHERE id = ?", (status, doc_id))
    conn.commit()


def _row_to_document(row) -> Document:
    return Document(
        id=row["id"], filename=row["filename"], sha256=row["sha256"],
        page_count=row["page_count"], title=row["title"], publisher=row["publisher"],
        doc_date=date.fromisoformat(row["doc_date"]) if row["doc_date"] else None,
        uploaded_at=datetime.fromisoformat(row["uploaded_at"]), status=row["status"],
    )


def get_document(conn, doc_id):
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return _row_to_document(row) if row else None


def get_document_by_sha256(conn, sha256):
    row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
    return _row_to_document(row) if row else None


def list_documents(conn):
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    return [_row_to_document(r) for r in rows]


def list_documents_with_claim_counts(conn):
    """Every documents row plus a claim_count, as plain dicts (spec Section 7).

    A LEFT JOIN so documents with zero claims -- still parsing, failed, or
    simply fact-free -- are not dropped from the listing.
    """
    rows = conn.execute(
        "SELECT documents.*, COUNT(claims.id) AS claim_count "
        "FROM documents LEFT JOIN claims ON claims.doc_id = documents.id "
        "GROUP BY documents.id ORDER BY documents.uploaded_at DESC"
    ).fetchall()
    out = []
    for row in rows:
        record = _row_to_document(row).__dict__
        record["claim_count"] = row["claim_count"]
        out.append(record)
    return out


# ---- chunks ----

def insert_chunks(conn, chunks):
    if not chunks:
        return
    conn.executemany(
        "INSERT INTO chunks (id, doc_id, page_no, kind, text, char_start, content_hash, score, extracted) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(c.id, c.doc_id, c.page_no, c.kind, c.text, c.char_start, c.content_hash, c.score, int(c.extracted))
         for c in chunks],
    )
    conn.commit()


def mark_chunk_extracted(conn, chunk_id):
    conn.execute("UPDATE chunks SET extracted = 1 WHERE id = ?", (chunk_id,))
    conn.commit()


def _row_to_chunk(row) -> Chunk:
    return Chunk(
        id=row["id"], doc_id=row["doc_id"], page_no=row["page_no"], kind=row["kind"],
        text=row["text"], char_start=row["char_start"], content_hash=row["content_hash"],
        score=row["score"], extracted=bool(row["extracted"]),
    )


def get_chunk(conn, chunk_id):
    row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    return _row_to_chunk(row) if row else None


# ---- claims ----

def _claim_to_params(c: Claim) -> tuple:
    return (
        c.id, c.doc_id, c.chunk_id, c.subject, c.subject_key, c.measure, c.measure_key,
        c.value_type, c.value_num, c.value_text, c.unit_raw, c.unit_dim, c.unit_scale,
        c.period_start.isoformat() if c.period_start else None,
        c.period_end.isoformat() if c.period_end else None,
        c.period_label, json.dumps(c.basis or {}), c.evidence_page, c.evidence_quote,
        c.evidence_char_start, c.confidence, c.extractor_version,
    )


def insert_claims(conn, claims):
    if not claims:
        return
    conn.executemany(
        "INSERT INTO claims (id, doc_id, chunk_id, subject, subject_key, measure, measure_key, "
        "value_type, value_num, value_text, unit_raw, unit_dim, unit_scale, period_start, "
        "period_end, period_label, basis, evidence_page, evidence_quote, evidence_char_start, "
        "confidence, extractor_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [_claim_to_params(c) for c in claims],
    )
    conn.commit()


def _row_to_claim(row, doc_vintage: date | None = None) -> Claim:
    return Claim(
        id=row["id"], doc_id=row["doc_id"], chunk_id=row["chunk_id"], subject=row["subject"],
        subject_key=row["subject_key"], measure=row["measure"], measure_key=row["measure_key"],
        value_type=row["value_type"], value_num=row["value_num"], value_text=row["value_text"],
        unit_raw=row["unit_raw"], unit_dim=row["unit_dim"], unit_scale=row["unit_scale"],
        period_start=date.fromisoformat(row["period_start"]) if row["period_start"] else None,
        period_end=date.fromisoformat(row["period_end"]) if row["period_end"] else None,
        period_label=row["period_label"], basis=json.loads(row["basis"] or "{}"),
        evidence_page=row["evidence_page"], evidence_quote=row["evidence_quote"],
        evidence_char_start=row["evidence_char_start"], confidence=row["confidence"],
        extractor_version=row["extractor_version"], doc_vintage=doc_vintage,
    )


def get_claim(conn, claim_id):
    row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
    return _row_to_claim(row) if row else None


def get_claims_for_document(conn, doc_id):
    rows = conn.execute("SELECT * FROM claims WHERE doc_id = ?", (doc_id,)).fetchall()
    return [_row_to_claim(r) for r in rows]


def get_all_claims_with_vintage(conn):
    rows = conn.execute(
        "SELECT claims.*, documents.doc_date AS doc_vintage FROM claims "
        "JOIN documents ON documents.id = claims.doc_id"
    ).fetchall()
    out = []
    for row in rows:
        vintage = date.fromisoformat(row["doc_vintage"]) if row["doc_vintage"] else None
        out.append(_row_to_claim(row, doc_vintage=vintage))
    return out


def list_claims(conn, doc_id=None, subject_key=None, measure_key=None, q=None):
    clauses, params = [], []
    if doc_id:
        clauses.append("doc_id = ?"); params.append(doc_id)
    if subject_key:
        clauses.append("subject_key = ?"); params.append(subject_key)
    if measure_key:
        clauses.append("measure_key = ?"); params.append(measure_key)
    if q:
        clauses.append("(subject LIKE ? OR measure LIKE ? OR evidence_quote LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(f"SELECT * FROM claims {where}", params).fetchall()
    return [_row_to_claim(r) for r in rows]


def list_groups(conn):
    rows = conn.execute(
        "SELECT subject_key, measure_key, COUNT(*) as claim_count FROM claims "
        "GROUP BY subject_key, measure_key ORDER BY claim_count DESC"
    ).fetchall()
    groups = []
    for row in rows:
        verdict_rows = conn.execute(
            "SELECT r.final_verdict, COUNT(*) as n FROM relations r "
            "JOIN claims a ON a.id = r.claim_a_id WHERE a.subject_key = ? AND a.measure_key = ? "
            "GROUP BY r.final_verdict",
            (row["subject_key"], row["measure_key"]),
        ).fetchall()
        groups.append({
            "subject_key": row["subject_key"], "measure_key": row["measure_key"],
            "claim_count": row["claim_count"],
            "verdict_counts": {v["final_verdict"]: v["n"] for v in verdict_rows},
        })
    return groups


def get_group(conn, subject_key, measure_key):
    claims = list_claims(conn, subject_key=subject_key, measure_key=measure_key)
    claim_ids = [c.id for c in claims]
    relations = []
    if claim_ids:
        placeholders = ",".join("?" * len(claim_ids))
        rows = conn.execute(
            f"SELECT * FROM relations WHERE claim_a_id IN ({placeholders}) OR claim_b_id IN ({placeholders})",
            claim_ids + claim_ids,
        ).fetchall()
        relations = [_row_to_relation(r) for r in rows]
    return {"claims": [c.__dict__ for c in claims], "relations": [r.__dict__ for r in relations]}


# ---- relations ----

def insert_relations(conn, relations):
    if not relations:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO relations (id, claim_a_id, claim_b_id, rule_verdict, final_verdict, "
        "axis, rule_fired, delta_pct, explanation, llm_overrode, override_reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r.id, r.claim_a_id, r.claim_b_id, r.rule_verdict, r.final_verdict, r.axis,
          r.rule_fired, r.delta_pct, r.explanation, int(r.llm_overrode), r.override_reason)
         for r in relations],
    )
    conn.commit()


def update_relation_explanation(conn, relation_id, final_verdict, explanation, llm_overrode, override_reason):
    """Persist the result of running explain_relation() over an already-inserted relation.

    Called once per relation right after explain_relation() mutates it, so
    each explanation is durably written before moving to the next one -- a
    mid-batch interrupt then only loses explanations not yet attempted, never
    relations that were already reconciled and inserted.
    """
    conn.execute(
        "UPDATE relations SET final_verdict = ?, explanation = ?, llm_overrode = ?, override_reason = ? "
        "WHERE id = ?",
        (final_verdict, explanation, int(llm_overrode), override_reason, relation_id),
    )
    conn.commit()


def get_relation_pairs_seen(conn) -> set:
    rows = conn.execute("SELECT claim_a_id, claim_b_id FROM relations").fetchall()
    return {(r["claim_a_id"], r["claim_b_id"]) for r in rows}


def _row_to_relation(row) -> Relation:
    return Relation(
        id=row["id"], claim_a_id=row["claim_a_id"], claim_b_id=row["claim_b_id"],
        rule_verdict=row["rule_verdict"], final_verdict=row["final_verdict"], axis=row["axis"],
        rule_fired=row["rule_fired"], delta_pct=row["delta_pct"], explanation=row["explanation"],
        llm_overrode=bool(row["llm_overrode"]), override_reason=row["override_reason"],
    )


def list_relations(conn, final_verdict=None, axis=None):
    clauses, params = [], []
    if final_verdict:
        clauses.append("final_verdict = ?"); params.append(final_verdict)
    if axis:
        clauses.append("axis = ?"); params.append(axis)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(f"SELECT * FROM relations {where}", params).fetchall()
    return [_row_to_relation(r) for r in rows]


# ---- failures ----

def insert_failures(conn, doc_id, failures):
    if not failures:
        return
    now = datetime.now().isoformat()
    conn.executemany(
        "INSERT INTO failures (id, doc_id, chunk_id, kind, reason, raw_payload, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(str(uuid.uuid4()), doc_id, f.get("chunk_id"), f.get("kind", "extraction"),
          f["reason"], json.dumps(f.get("raw", {})), now)
         for f in failures],
    )
    conn.commit()


def list_failures(conn):
    rows = conn.execute("SELECT * FROM failures ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]
