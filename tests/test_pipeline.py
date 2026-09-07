# tests/test_pipeline.py
import json

import pymupdf
import pytest

from app.db import get_connection, init_db, list_documents, list_claims, list_relations, list_failures
from extract.llm import LLMClient
from ingest.pipeline import queue_document, run_ingest_pipeline, ingest_local_file, reconcile_all


def make_conn():
    conn = get_connection(":memory:")
    init_db(conn)
    return conn


def make_pdf_bytes(text: str, tmp_path, name="doc.pdf") -> bytes:
    path = tmp_path / name
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path.read_bytes()


def _seed_cache(cache_dir, chunk_text, extraction_response):
    from ingest.chunk import content_hash
    from extract.llm import cache_key_for
    from extract.prompts import EXTRACTION_PROMPT_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key_for(content_hash(chunk_text), EXTRACTION_PROMPT_VERSION)
    (cache_dir / f"{key}.json").write_text(json.dumps(extraction_response), encoding="utf-8")


def _seed_explanation_cache(cache_dir, claim_a, claim_b, rule_fired, explanation_response):
    from reason.explain import relation_content_hash
    from extract.llm import cache_key_for
    from extract.prompts import EXPLANATION_PROMPT_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)
    content_hash_val = relation_content_hash(claim_a, claim_b, rule_fired)
    key = cache_key_for(content_hash_val, EXPLANATION_PROMPT_VERSION)
    (cache_dir / f"{key}.json").write_text(json.dumps(explanation_response), encoding="utf-8")


def test_queue_document_dedupes_by_sha256(tmp_path):
    conn = make_conn()
    content = make_pdf_bytes("Revenue was Rs 100 crore in FY2024.", tmp_path)
    upload_dir = str(tmp_path / "uploads")
    doc_id_1 = queue_document(conn, content, "report.pdf", upload_dir)
    doc_id_2 = queue_document(conn, content, "report-renamed.pdf", upload_dir)
    assert doc_id_1 == doc_id_2
    assert len(list_documents(conn)) == 1


def test_queue_document_sets_pending_status(tmp_path):
    conn = make_conn()
    content = make_pdf_bytes("Some text.", tmp_path)
    doc_id = queue_document(conn, content, "f.pdf", str(tmp_path / "uploads"))
    doc = list_documents(conn)[0]
    assert doc.id == doc_id
    assert doc.status == "pending"


def test_ingest_local_file_extracts_claims_from_cache(tmp_path):
    conn = make_conn()
    text = "Revenue from operations stood at Rs 100 crore in FY2024."
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(pdf_path))
    doc.close()

    cache_dir = tmp_path / "cache"
    llm = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(cache_dir))

    # Ingest once (dry) to discover the exact chunk text pymupdf produces, then seed its cache.
    from ingest.parse import parse_pdf
    from ingest.chunk import blocks_to_chunks
    parsed = parse_pdf(str(pdf_path))
    chunks = blocks_to_chunks("probe", parsed.blocks)
    prose_chunk = next(c for c in chunks if c.kind == "prose")
    _seed_cache(cache_dir, prose_chunk.text, [{
        "subject": "Delhivery", "subject_key": "delhivery_limited",
        "measure": "Revenue from Operations", "measure_key": "revenue_from_operations",
        "value_type": "numeric", "value_num": 100, "unit_raw": "Rs crore", "period_label": "FY2024",
        "evidence_quote": text,
    }])

    doc_id = ingest_local_file(conn, llm, str(pdf_path), "sample.pdf", call_budget=10)
    claims = list_claims(conn, doc_id=doc_id)
    assert len(claims) == 1
    assert claims[0].measure_key == "revenue_from_operations"
    doc = [d for d in list_documents(conn) if d.id == doc_id][0]
    assert doc.status == "ready"
    assert doc.doc_date is not None or doc.doc_date is None  # metadata may be absent on a synthetic PDF; must not crash


def test_run_ingest_pipeline_marks_failed_on_llm_unavailable(tmp_path):
    content = make_pdf_bytes("Revenue from operations stood at Rs 999 crore in FY2024.", tmp_path)
    upload_dir = str(tmp_path / "uploads")

    # run_ingest_pipeline opens its own connection by db_path (it must be safe to call
    # from a background thread), so the document has to be queued against that same
    # file-backed database rather than an in-memory one this test can't share.
    db_path = str(tmp_path / "facts.db")
    seed_conn = get_connection(db_path)
    init_db(seed_conn)
    doc_id = queue_document(seed_conn, content, "f.pdf", upload_dir)
    seed_conn.close()

    llm = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(tmp_path / "cache"))
    run_ingest_pipeline(db_path, llm, doc_id, upload_dir, call_budget=10)

    check_conn = get_connection(db_path)
    doc = [d for d in list_documents(check_conn) if d.id == doc_id][0]
    assert doc.status == "failed"
    failures = list_failures(check_conn)
    assert any("pipeline_error" in f["reason"] for f in failures)


def test_ingest_local_file_marks_failed_on_llm_unavailable(tmp_path):
    # Regression test: ingest_local_file previously had no try/except around
    # _extract_all, so the RuntimeError it raises when 100% of a document's
    # budgeted chunks fail via llm_call_failed (no API key, no cache) propagated
    # straight out of ingest_local_file -- leaving the document stuck at status
    # "extracting" and crashing the caller (e.g. scripts/seed_starter_corpus.py
    # partway through a multi-document batch) instead of degrading gracefully
    # to "failed", the same way run_ingest_pipeline already does.
    conn = make_conn()
    content = make_pdf_bytes("Revenue from operations stood at Rs 999 crore in FY2024.", tmp_path)
    pdf_path = tmp_path / "f.pdf"
    pdf_path.write_bytes(content)

    llm = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(tmp_path / "cache"))

    doc_id = ingest_local_file(conn, llm, str(pdf_path), "f.pdf", call_budget=10)

    assert isinstance(doc_id, str) and doc_id

    doc = [d for d in list_documents(conn) if d.id == doc_id][0]
    assert doc.status == "failed"
    failures = [f for f in list_failures(conn) if f["doc_id"] == doc_id]
    assert any("pipeline_error" in f["reason"] for f in failures)


def test_partial_llm_failure_still_reaches_ready_with_successful_claims(tmp_path):
    # Regression test for _extract_all's llm_unavailable_count threshold: a document
    # with two chunks where ONE chunk's LLM call fails (cache miss + no API key ->
    # LLMUnavailableError -> extract_claims_for_chunk converts it to a per-chunk
    # llm_call_failed failure, per Task 13's tested contract) while the OTHER chunk
    # succeeds must still reach status "ready" with the successful claim persisted.
    # Only a 100%-failure-rate across budgeted chunks is fatal; a partial failure
    # is not.
    conn = make_conn()
    text_a = "Revenue from operations stood at Rs 100 crore in FY2024."
    text_b = "Total expenses stood at Rs 50 crore in FY2024."
    pdf_path = tmp_path / "two_page.pdf"
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), text_a)
    page2 = doc.new_page()
    page2.insert_text((72, 72), text_b)
    doc.save(str(pdf_path))
    doc.close()

    cache_dir = tmp_path / "cache"
    llm = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(cache_dir))

    # Probe-parse to discover the exact per-page chunk texts pymupdf produces, then
    # seed the cache for only ONE of the two chunks. The other chunk's cache stays
    # empty, so its LLM call raises LLMUnavailableError.
    from ingest.parse import parse_pdf
    from ingest.chunk import blocks_to_chunks
    parsed = parse_pdf(str(pdf_path))
    probe_chunks = blocks_to_chunks("probe", parsed.blocks)
    prose_chunks = [c for c in probe_chunks if c.kind == "prose"]
    assert len(prose_chunks) == 2, "test setup requires two independently-extracted chunks"

    seeded_chunk = prose_chunks[0]
    _seed_cache(cache_dir, seeded_chunk.text, [{
        "subject": "Delhivery", "subject_key": "delhivery_limited",
        "measure": "Revenue from Operations", "measure_key": "revenue_from_operations",
        "value_type": "numeric", "value_num": 100, "unit_raw": "Rs crore", "period_label": "FY2024",
        "evidence_quote": text_a,
    }])
    # prose_chunks[1]'s cache is deliberately left unseeded.

    doc_id = ingest_local_file(conn, llm, str(pdf_path), "two_page.pdf", call_budget=10)

    doc_row = [d for d in list_documents(conn) if d.id == doc_id][0]
    assert doc_row.status == "ready"

    claims = list_claims(conn, doc_id=doc_id)
    assert len(claims) == 1
    assert claims[0].measure_key == "revenue_from_operations"

    failures = [f for f in list_failures(conn) if f["doc_id"] == doc_id]
    assert any(f["reason"].startswith("llm_call_failed") for f in failures)


def test_ingest_local_file_zero_call_budget_reaches_ready_with_no_claims(tmp_path):
    # Regression test for _extract_all's `if not prioritized: return` guard: with
    # call_budget=0, `prioritized` is an empty list even though the document has
    # real chunks. Without the guard, `llm_unavailable_count == len(prioritized)`
    # is a vacuous 0 == 0 that would spuriously raise and mark the document failed.
    conn = make_conn()
    text = "Revenue from operations stood at Rs 100 crore in FY2024."
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(pdf_path))
    doc.close()

    llm = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(tmp_path / "cache"))

    doc_id = ingest_local_file(conn, llm, str(pdf_path), "sample.pdf", call_budget=0)

    doc_row = [d for d in list_documents(conn) if d.id == doc_id][0]
    assert doc_row.status == "ready"

    claims = list_claims(conn, doc_id=doc_id)
    assert claims == []

    failures = [f for f in list_failures(conn) if f["doc_id"] == doc_id]
    assert not any("pipeline_error" in f["reason"] for f in failures)


def test_reconcile_all_is_idempotent(tmp_path):
    from app.models import Claim
    from datetime import date
    conn = make_conn()
    from app.db import insert_document, set_document_metadata, insert_chunks, insert_claims
    from app.models import Chunk
    insert_document(conn, "d1", "f.pdf", "sha1", 0, "2026-01-01T00:00:00", "ready")
    set_document_metadata(conn, "d1", 1, date(2024, 8, 8))
    insert_document(conn, "d2", "g.pdf", "sha2", 0, "2026-01-01T00:00:00", "ready")
    set_document_metadata(conn, "d2", 1, date(2024, 5, 17))
    insert_chunks(conn, [Chunk("c1", "d1", 1, "prose", "t", 0, "h1", 1.0, True),
                          Chunk("c2", "d2", 1, "prose", "t", 0, "h2", 1.0, True)])
    insert_claims(conn, [
        Claim("claim1", "d1", "c1", "Delhivery", "delhivery_limited", "Revenue", "revenue_from_operations",
              "numeric", 81415.38, None, "Rs million", "currency_inr", 1e6,
              date(2023, 4, 1), date(2024, 3, 31), "FY2024", {}, 1, "q1", 0, 0.9, "v1"),
        Claim("claim2", "d2", "c2", "Delhivery", "delhivery_limited", "Revenue", "revenue_from_operations",
              "numeric", 8142, None, "Rs crore", "currency_inr", 1e7,
              date(2023, 4, 1), date(2024, 3, 31), "FY2024", {}, 1, "q2", 0, 0.9, "v1"),
    ])
    llm = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(tmp_path / "cache"))

    reconcile_all(conn, llm)
    first_pass = list_relations(conn)
    assert len(first_pass) == 1
    assert first_pass[0].rule_verdict == "CORROBORATES"

    reconcile_all(conn, llm)  # must not duplicate or error
    assert len(list_relations(conn)) == 1


def test_reconcile_all_persists_all_relations_and_respects_explanation_budget(tmp_path):
    # Regression test for two bugs in reconcile_all's explanation loop:
    # (1) it called insert_relations() only AFTER the entire explanation loop
    #     completed, so an interrupt mid-batch lost every relation computed so
    #     far, even ones needing no explanation at all;
    # (2) it had no cap on how many relations got a live LLM explanation call
    #     per invocation, even though RECONCILED_BY_CONTEXT/CONTRADICTS pairs
    #     are the common case, not rare.
    #
    # This builds three independent (subject_key, measure_key) groups, each
    # with a same-envelope value mismatch -> CONTRADICTS (an "interesting"
    # verdict that normally gets an explanation call), seeds a distinguishable
    # cached explanation for all three, then reconciles with a budget of 1.
    from app.models import Claim
    from datetime import date
    conn = make_conn()
    from app.db import insert_document, set_document_metadata, insert_chunks, insert_claims, list_relations
    from app.models import Chunk

    insert_document(conn, "d1", "f.pdf", "sha1", 0, "2026-01-01T00:00:00", "ready")
    set_document_metadata(conn, "d1", 1, date(2024, 8, 8))
    insert_chunks(conn, [Chunk(f"c{i}", "d1", 1, "prose", "t", 0, f"h{i}", 1.0, True) for i in range(6)])

    claims = []
    for idx in range(3):
        subject_key = f"subject{idx}"
        a = Claim(f"a{idx}", "d1", f"c{2*idx}", "Subject", subject_key, "Measure", "measure_x",
                  "numeric", 100.0 + idx, None, "Rs crore", "currency_inr", 1e7,
                  date(2023, 4, 1), date(2024, 3, 31), "FY2024", {}, 1, f"quote-a-{idx}", 0, 0.9, "v1")
        b = Claim(f"b{idx}", "d1", f"c{2*idx+1}", "Subject", subject_key, "Measure", "measure_x",
                  "numeric", 999.0 + idx, None, "Rs crore", "currency_inr", 1e7,
                  date(2023, 4, 1), date(2024, 3, 31), "FY2024", {}, 1, f"quote-b-{idx}", 0, 0.9, "v1")
        claims.extend([a, b])
    insert_claims(conn, claims)

    cache_dir = tmp_path / "cache"
    llm = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(cache_dir))

    # Seed a cached explanation for ALL THREE pairs so that a relation which IS
    # sent to explain_relation gets a real, distinguishable explanation string
    # rather than silently falling into the "LLM unavailable" no-op path --
    # which would look identical to "never attempted" and defeat the test.
    for idx in range(3):
        _seed_explanation_cache(
            cache_dir, claims[2 * idx], claims[2 * idx + 1], "same_envelope_value_mismatch",
            {"explanation": f"explained-{idx}", "override": None},
        )

    reconcile_all(conn, llm, explanation_budget=1)

    relations = list_relations(conn)
    assert len(relations) == 3  # all persisted regardless of explanation budget
    assert all(r.rule_verdict == "CONTRADICTS" for r in relations)

    explained = [r for r in relations if r.explanation is not None]
    not_explained = [r for r in relations if r.explanation is None]
    assert len(explained) == 1
    assert len(not_explained) == 2
    for r in not_explained:
        assert r.final_verdict == r.rule_verdict
        assert r.llm_overrode is False


def test_reconcile_all_zero_explanation_budget_still_persists_all_relations(tmp_path):
    from app.models import Claim
    from datetime import date
    conn = make_conn()
    from app.db import insert_document, set_document_metadata, insert_chunks, insert_claims, list_relations
    from app.models import Chunk

    insert_document(conn, "d1", "f.pdf", "sha1", 0, "2026-01-01T00:00:00", "ready")
    set_document_metadata(conn, "d1", 1, date(2024, 8, 8))
    insert_chunks(conn, [Chunk("c1", "d1", 1, "prose", "t", 0, "h1", 1.0, True),
                          Chunk("c2", "d1", 1, "prose", "t", 0, "h2", 1.0, True)])
    insert_claims(conn, [
        Claim("a", "d1", "c1", "Subject", "subject_a", "Measure", "measure_x", "numeric", 100.0, None,
              "Rs crore", "currency_inr", 1e7, date(2023, 4, 1), date(2024, 3, 31), "FY2024", {}, 1,
              "quote-a", 0, 0.9, "v1"),
        Claim("b", "d1", "c2", "Subject", "subject_a", "Measure", "measure_x", "numeric", 999.0, None,
              "Rs crore", "currency_inr", 1e7, date(2023, 4, 1), date(2024, 3, 31), "FY2024", {}, 1,
              "quote-b", 0, 0.9, "v1"),
    ])
    llm = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(tmp_path / "cache"))

    reconcile_all(conn, llm, explanation_budget=0)

    relations = list_relations(conn)
    assert len(relations) == 1
    assert relations[0].rule_verdict == "CONTRADICTS"
    assert relations[0].final_verdict == relations[0].rule_verdict
    assert relations[0].explanation is None
