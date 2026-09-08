# app/main.py
import json
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db import (
    get_claim, get_connection, get_document, get_group, init_db, list_claims,
    list_documents_with_claim_counts, list_failures, list_groups, list_relations,
)
from extract.llm import LLMClient
from ingest.pipeline import queue_document, reconcile_all, run_ingest_pipeline

settings = get_settings()
app = FastAPI(title="Fact Knowledge Layer")
UPLOAD_DIR = "data/uploads"
STATIC_DIR = Path(__file__).parent / "static"


def _conn():
    conn = get_connection(settings.db_path)
    init_db(conn)
    return conn


def _llm() -> LLMClient:
    return LLMClient(settings.gemini_api_key, settings.gemini_model, settings.cache_dir)


@app.post("/api/documents")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are supported")
    content = await file.read()
    conn = _conn()
    doc_id = queue_document(conn, content, file.filename, UPLOAD_DIR)
    conn.close()
    background_tasks.add_task(
        run_ingest_pipeline, settings.db_path, _llm(), doc_id, UPLOAD_DIR,
        settings.call_budget_per_document, settings.explanation_budget_per_call,
    )
    return {"doc_id": doc_id}


@app.get("/api/documents")
def get_documents():
    conn = _conn()
    docs = list_documents_with_claim_counts(conn)
    conn.close()
    return docs


@app.get("/api/documents/{doc_id}")
def get_document_detail(doc_id: str):
    conn = _conn()
    doc = get_document(conn, doc_id)
    conn.close()
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc.__dict__


@app.get("/api/claims")
def get_claims(doc_id: str | None = None, subject_key: str | None = None,
               measure_key: str | None = None, q: str | None = None):
    conn = _conn()
    claims = list_claims(conn, doc_id=doc_id, subject_key=subject_key, measure_key=measure_key, q=q)
    conn.close()
    return [c.__dict__ for c in claims]


@app.get("/api/groups")
def get_groups():
    conn = _conn()
    groups = list_groups(conn)
    conn.close()
    return groups


@app.get("/api/groups/{subject_key}/{measure_key}")
def get_group_detail(subject_key: str, measure_key: str):
    conn = _conn()
    group = get_group(conn, subject_key, measure_key)
    conn.close()
    return group


@app.get("/api/relations")
def get_relations(final_verdict: str | None = None, axis: str | None = None):
    conn = _conn()
    relations = list_relations(conn, final_verdict=final_verdict, axis=axis)
    conn.close()
    return [r.__dict__ for r in relations]


@app.get("/api/evidence/{claim_id}")
def get_evidence(claim_id: str):
    conn = _conn()
    claim = get_claim(conn, claim_id)
    doc = get_document(conn, claim.doc_id) if claim else None
    conn.close()
    if claim is None:
        raise HTTPException(404, "claim not found")
    start = claim.evidence_char_start or 0
    end = start + len(claim.evidence_quote)
    return {
        "claim_id": claim.id, "page": claim.evidence_page, "quote": claim.evidence_quote,
        "char_start": start, "char_end": end,
        "document": doc.filename if doc else None,
        "subject": claim.subject, "measure": claim.measure,
        "value": claim.value_num if claim.value_type == "numeric" else claim.value_text,
        "unit": claim.unit_raw, "period": claim.period_label,
        "basis": claim.basis, "vintage": doc.doc_date.isoformat() if doc and doc.doc_date else None,
    }


@app.get("/api/failures")
def get_failures():
    conn = _conn()
    failures = list_failures(conn)
    conn.close()
    # app/db.py's list_failures() returns raw_payload as a raw JSON string (unlike
    # claims.basis, which db.py itself decodes into a dict before returning). We
    # decode it here at the API boundary so every JSON-shaped field in the API
    # response is consistently a real JSON object, not a double-encoded string
    # the frontend would have to JSON.parse() again.
    for failure in failures:
        raw_payload = failure.get("raw_payload")
        if raw_payload:
            try:
                failure["raw_payload"] = json.loads(raw_payload)
            except (TypeError, ValueError):
                pass
    return failures


@app.post("/api/reconcile")
def post_reconcile():
    conn = _conn()
    try:
        reconcile_all(conn, _llm(), explanation_budget=settings.explanation_budget_per_call)
    finally:
        conn.close()
    return {"status": "reconciled"}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
