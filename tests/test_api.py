import time

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import db as db_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "facts.db")
    monkeypatch.setenv("FKL_DB_PATH", db_path)
    monkeypatch.setenv("FKL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import app.main as main_module
    import importlib
    importlib.reload(main_module)
    main_module.UPLOAD_DIR = str(tmp_path / "uploads")

    with TestClient(main_module.app) as test_client:
        yield test_client, db_path


def _seed_sample_data(db_path):
    from datetime import date
    from app.db import get_connection, init_db, insert_document, set_document_metadata, insert_chunks, insert_claims, insert_relations
    from app.models import Chunk, Claim, Relation

    conn = get_connection(db_path)
    init_db(conn)
    insert_document(conn, "d1", "annual_report.pdf", "sha1", 0, "2026-01-01T00:00:00", "ready")
    set_document_metadata(conn, "d1", 10, date(2024, 8, 8))
    insert_chunks(conn, [Chunk("c1", "d1", 5, "prose", "Revenue was Rs 8142 crore.", 0, "h1", 5.0, True)])
    insert_claims(conn, [Claim(
        "claim1", "d1", "c1", "Delhivery", "delhivery_limited", "Revenue", "revenue_from_operations",
        "numeric", 8142.0, None, "Rs crore", "currency_inr", 1e7,
        date(2023, 4, 1), date(2024, 3, 31), "FY2024", {}, 5, "Revenue was Rs 8142 crore.", 12,
        0.9, "v1",
    )])
    insert_relations(conn, [Relation(
        "r1", "claim1", "claim1b", "CORROBORATES", "CORROBORATES", None, "unit_normalized_match", 0.0001,
    )])
    conn.close()
    return conn


def test_get_documents_empty_initially(client):
    test_client, db_path = client
    response = test_client.get("/api/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_get_documents_includes_per_document_claim_count(client):
    test_client, db_path = client
    _seed_sample_data(db_path)
    response = test_client.get("/api/documents")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["claim_count"] == 1
    assert body[0]["filename"] == "annual_report.pdf"


def test_get_claims_returns_seeded_claim(client):
    test_client, db_path = client
    _seed_sample_data(db_path)
    response = test_client.get("/api/claims", params={"measure_key": "revenue_from_operations"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["subject_key"] == "delhivery_limited"


def test_get_relations_filters_by_final_verdict(client):
    test_client, db_path = client
    _seed_sample_data(db_path)
    response = test_client.get("/api/relations", params={"final_verdict": "CORROBORATES"})
    assert response.status_code == 200
    assert len(response.json()) == 1
    response_empty = test_client.get("/api/relations", params={"final_verdict": "CONTRADICTS"})
    assert response_empty.json() == []


def test_get_groups(client):
    test_client, db_path = client
    _seed_sample_data(db_path)
    response = test_client.get("/api/groups")
    assert response.status_code == 200
    assert response.json()[0]["subject_key"] == "delhivery_limited"


def test_get_evidence_for_claim(client):
    test_client, db_path = client
    _seed_sample_data(db_path)
    response = test_client.get("/api/evidence/claim1")
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 5
    assert "Rs 8142 crore" in body["quote"]


def test_get_evidence_404_for_missing_claim(client):
    test_client, db_path = client
    response = test_client.get("/api/evidence/does-not-exist")
    assert response.status_code == 404


def test_get_failures_empty_initially(client):
    test_client, db_path = client
    response = test_client.get("/api/failures")
    assert response.status_code == 200
    assert response.json() == []


def test_upload_rejects_non_pdf(client):
    test_client, db_path = client
    response = test_client.post("/api/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_upload_rejects_none_filename_without_crashing(client):
    # Regression test: file.filename.lower() raised AttributeError (-> uncaught
    # 500) when file.filename is None. Standard multipart parsing in this
    # starlette version can't actually produce filename=None over real HTTP
    # (a part with no "filename" option in Content-Disposition is parsed as a
    # plain form field, not an UploadFile, and FastAPI rejects it with 422
    # before our handler runs) -- but UploadFile's own type is `str | None`,
    # so a well-behaved handler must not crash on that input regardless. This
    # calls the endpoint function directly with a manually constructed
    # UploadFile(filename=None) to exercise exactly that guard.
    import io

    import anyio
    from fastapi import BackgroundTasks, HTTPException, UploadFile

    import app.main as main_module

    test_client, db_path = client  # ensures app.main is reloaded against this test's settings

    async def _run():
        upload = UploadFile(file=io.BytesIO(b"hello"), filename=None)
        with pytest.raises(HTTPException) as exc_info:
            await main_module.upload_document(upload, BackgroundTasks())
        return exc_info.value

    exc = anyio.run(_run)
    assert exc.status_code == 400


def test_upload_returns_doc_id_and_eventually_settles_status(client):
    test_client, db_path = client
    pdf_bytes_io = _make_tiny_pdf_bytes("Revenue was Rs 100 crore in FY2024.")
    response = test_client.post("/api/documents", files={"file": ("report.pdf", pdf_bytes_io, "application/pdf")})
    assert response.status_code == 200
    doc_id = response.json()["doc_id"]

    detail = None
    for _ in range(10):
        detail_response = test_client.get(f"/api/documents/{doc_id}")
        if detail_response.status_code == 200:
            detail = detail_response.json()
            if detail["status"] in ("ready", "failed"):
                break
        time.sleep(0.2)
    assert detail is not None
    # No GEMINI_API_KEY and no pre-seeded cache in this test environment -> "failed" is the
    # expected, honest outcome (spec Section 10: LLM output quality is not asserted in tests).
    assert detail["status"] in ("ready", "failed")


def _make_tiny_pdf_bytes(text: str) -> bytes:
    import io
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buf = io.BytesIO(doc.tobytes())
    doc.close()
    return buf.getvalue()
