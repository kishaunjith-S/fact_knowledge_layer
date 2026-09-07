# tests/test_db.py
from datetime import date, datetime

from app.db import (
    get_connection, init_db, insert_document, set_document_metadata,
    update_document_status, get_document, get_document_by_sha256, list_documents,
    list_documents_with_claim_counts,
    insert_chunks, mark_chunk_extracted, get_chunk,
    insert_claims, get_claim, get_claims_for_document, get_all_claims_with_vintage,
    list_claims, list_groups, get_group,
    insert_relations, get_relation_pairs_seen, list_relations,
    update_relation_explanation,
    insert_failures, list_failures,
)
from app.models import Chunk, Claim, Relation


def make_conn():
    conn = get_connection(":memory:")
    init_db(conn)
    return conn


def test_document_roundtrip():
    conn = make_conn()
    insert_document(conn, "d1", "report.pdf", "sha-abc", page_count=0,
                     uploaded_at="2026-01-01T00:00:00+00:00", status="pending")
    set_document_metadata(conn, "d1", page_count=42, doc_date=date(2025, 5, 25))
    update_document_status(conn, "d1", "ready")

    doc = get_document(conn, "d1")
    assert doc.filename == "report.pdf"
    assert doc.page_count == 42
    assert doc.doc_date == date(2025, 5, 25)
    assert doc.status == "ready"
    assert get_document_by_sha256(conn, "sha-abc").id == "d1"
    assert len(list_documents(conn)) == 1


def test_list_documents_with_claim_counts():
    conn = make_conn()
    insert_document(conn, "d1", "a.pdf", "sha1", 1, "2026-01-02T00:00:00", "ready")
    insert_document(conn, "d2", "b.pdf", "sha2", 1, "2026-01-01T00:00:00", "ready")
    insert_chunks(conn, [Chunk("c1", "d1", 1, "prose", "t", 0, "h1", 1.0, False)])

    def _claim(cid, doc_id):
        return Claim(
            id=cid, doc_id=doc_id, chunk_id="c1", subject="S", subject_key="s",
            measure="M", measure_key="m", value_type="numeric", value_num=1.0, value_text=None,
            unit_raw=None, unit_dim=None, unit_scale=1.0, period_start=None, period_end=None,
            period_label=None, basis={}, evidence_page=1, evidence_quote="t", evidence_char_start=0,
            confidence=1.0, extractor_version="v1",
        )

    insert_claims(conn, [_claim("x1", "d1"), _claim("x2", "d1")])

    rows = list_documents_with_claim_counts(conn)
    by_id = {r["id"]: r for r in rows}
    assert by_id["d1"]["claim_count"] == 2
    assert by_id["d2"]["claim_count"] == 0          # documents with no claims still appear
    assert by_id["d1"]["filename"] == "a.pdf"       # every documents column is still present
    assert rows[0]["id"] == "d1"                    # newest first, matching list_documents order


def test_chunk_roundtrip():
    conn = make_conn()
    insert_document(conn, "d1", "f.pdf", "sha1", 0, "2026-01-01T00:00:00", "pending")
    chunk = Chunk(id="c1", doc_id="d1", page_no=3, kind="table", text="Revenue: 100",
                  char_start=0, content_hash="hash1", score=5.0, extracted=False)
    insert_chunks(conn, [chunk])
    mark_chunk_extracted(conn, "c1")

    fetched = get_chunk(conn, "c1")
    assert fetched.page_no == 3
    assert fetched.extracted is True


def test_claim_roundtrip_and_group_queries():
    conn = make_conn()
    insert_document(conn, "d1", "f.pdf", "sha1", 0, "2026-01-01T00:00:00", "pending")
    set_document_metadata(conn, "d1", page_count=1, doc_date=date(2025, 5, 25))
    insert_chunks(conn, [Chunk("c1", "d1", 1, "prose", "text", 0, "h1", 1.0, False)])

    claim = Claim(
        id="claim1", doc_id="d1", chunk_id="c1", subject="RBI", subject_key="rbi",
        measure="Real GDP Growth", measure_key="real_gdp_growth", value_type="numeric",
        value_num=6.5, value_text=None, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
        period_start=date(2024, 4, 1), period_end=date(2025, 3, 31), period_label="2024-25",
        basis={}, evidence_page=5, evidence_quote="grew by 6.5 per cent", evidence_char_start=10,
        confidence=0.9, extractor_version="v1",
    )
    insert_claims(conn, [claim])

    fetched = get_claim(conn, "claim1")
    assert fetched.value_num == 6.5
    assert fetched.period_start == date(2024, 4, 1)
    assert fetched.basis == {}

    by_doc = get_claims_for_document(conn, "d1")
    assert len(by_doc) == 1

    with_vintage = get_all_claims_with_vintage(conn)
    assert with_vintage[0].doc_vintage == date(2025, 5, 25)

    filtered = list_claims(conn, measure_key="real_gdp_growth")
    assert len(filtered) == 1
    filtered_q = list_claims(conn, q="GDP")
    assert len(filtered_q) == 1

    groups = list_groups(conn)
    assert groups[0]["subject_key"] == "rbi"
    assert groups[0]["claim_count"] == 1

    group_detail = get_group(conn, "rbi", "real_gdp_growth")
    assert len(group_detail["claims"]) == 1


def test_relation_roundtrip_and_dedup():
    conn = make_conn()
    insert_document(conn, "d1", "f.pdf", "sha1", 0, "2026-01-01T00:00:00", "pending")
    relation = Relation(
        id="r1", claim_a_id="claimA", claim_b_id="claimB",
        rule_verdict="CONTRADICTS", final_verdict="CONTRADICTS", axis="vintage",
        rule_fired="large_gap_despite_vintage_gap", delta_pct=0.54,
    )
    insert_relations(conn, [relation])
    seen = get_relation_pairs_seen(conn)
    assert ("claimA", "claimB") in seen

    fetched = list_relations(conn, final_verdict="CONTRADICTS")
    assert len(fetched) == 1
    assert fetched[0].axis == "vintage"

    insert_relations(conn, [relation])  # duplicate insert must not raise or double-count
    assert len(list_relations(conn)) == 1


def test_update_relation_explanation_roundtrip():
    conn = make_conn()
    insert_document(conn, "d1", "f.pdf", "sha1", 0, "2026-01-01T00:00:00", "pending")
    relation = Relation(
        id="r1", claim_a_id="claimA", claim_b_id="claimB",
        rule_verdict="CONTRADICTS", final_verdict="CONTRADICTS", axis=None,
        rule_fired="same_envelope_value_mismatch", delta_pct=0.9,
    )
    insert_relations(conn, [relation])

    update_relation_explanation(
        conn, "r1", final_verdict="RECONCILED_BY_CONTEXT", explanation="the model's reasoning",
        llm_overrode=True, override_reason="context makes this consistent",
    )

    fetched = list_relations(conn)[0]
    assert fetched.final_verdict == "RECONCILED_BY_CONTEXT"
    assert fetched.explanation == "the model's reasoning"
    assert fetched.llm_overrode is True
    assert fetched.override_reason == "context makes this consistent"
    # rule_verdict must never be overwritten by an explanation update.
    assert fetched.rule_verdict == "CONTRADICTS"


def test_failures_roundtrip():
    conn = make_conn()
    insert_document(conn, "d1", "f.pdf", "sha1", 0, "2026-01-01T00:00:00", "pending")
    insert_failures(conn, "d1", [{"reason": "evidence_quote_not_found_in_chunk", "raw": {"subject": "X"}}])
    failures = list_failures(conn)
    assert len(failures) == 1
    assert failures[0]["reason"] == "evidence_quote_not_found_in_chunk"
