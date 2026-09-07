# ingest/pipeline.py
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.db import (
    get_connection, get_document_by_sha256, insert_document, set_document_metadata,
    update_document_status, insert_chunks, mark_chunk_extracted, insert_claims,
    insert_failures, get_all_claims_with_vintage, get_relation_pairs_seen, insert_relations,
    update_relation_explanation,
)
from extract.claims import extract_claims_for_chunk
from extract.llm import LLMClient
from ingest.chunk import blocks_to_chunks
from ingest.parse import parse_pdf
from ingest.score import score_chunk
from reason.align import group_claims
from reason.explain import INTERESTING_VERDICTS, explain_relation
from reason.reconcile import reconcile


def queue_document(conn, file_bytes: bytes, filename: str, upload_dir: str) -> str:
    """Register an uploaded PDF, deduped by content hash. Does NOT run the pipeline."""
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    existing = get_document_by_sha256(conn, sha256)
    if existing:
        return existing.id

    doc_id = str(uuid.uuid4())
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    (Path(upload_dir) / f"{doc_id}.pdf").write_bytes(file_bytes)
    insert_document(
        conn, doc_id, filename, sha256, page_count=0,
        uploaded_at=datetime.now(timezone.utc).isoformat(), status="pending",
    )
    return doc_id


def _extract_all(conn, llm: LLMClient, doc_id: str, chunks, call_budget: int) -> None:
    """Run extraction over the highest-scoring chunks, up to call_budget LLM calls.

    extract_claims_for_chunk() never raises on its own -- an LLM call failure
    (including LLMUnavailableError when there's no cache and no API key) is
    captured as a per-chunk failure record, not an exception (see
    extract/claims.py and tests/test_claims.py::test_llm_call_failure_is_captured_not_raised).
    That per-chunk resilience is correct for isolated/transient failures, but it
    means a document whose extraction failed on *every single* chunk because the
    LLM is entirely unavailable would otherwise sail through to "ready" with zero
    claims and no visible error. We treat that all-chunks-failed case as fatal so
    run_ingest_pipeline can mark the document "failed" instead of silently
    producing an empty result.
    """
    prioritized = sorted(chunks, key=lambda c: c.score, reverse=True)[:call_budget]
    if not prioritized:
        return

    llm_unavailable_count = 0
    for chunk in prioritized:
        claims, failures = extract_claims_for_chunk(conn, llm, chunk)
        insert_claims(conn, claims)
        insert_failures(conn, doc_id, failures)
        mark_chunk_extracted(conn, chunk.id)
        if not claims and any(f["reason"].startswith("llm_call_failed") for f in failures):
            llm_unavailable_count += 1

    if llm_unavailable_count == len(prioritized):
        raise RuntimeError(
            f"LLM unavailable: extraction failed for all {len(prioritized)} prioritized chunk(s)"
        )


def run_ingest_pipeline(
    db_path: str, llm: LLMClient, doc_id: str, upload_dir: str, call_budget: int,
    explanation_budget: int | None = None,
) -> None:
    """Run parse -> chunk+score -> extract -> align -> reconcile -> explain for one document.

    Opens its own SQLite connection by db_path (never a shared connection object)
    so it is safe to invoke from a background thread, e.g. FastAPI BackgroundTasks.
    Any exception during the run marks the document "failed" and logs the error
    into the failures table instead of raising out of the background task.
    """
    conn = get_connection(db_path)
    try:
        file_path = Path(upload_dir) / f"{doc_id}.pdf"
        update_document_status(conn, doc_id, "parsing")
        parsed = parse_pdf(str(file_path))
        set_document_metadata(conn, doc_id, parsed.page_count, parsed.doc_date)

        chunks = blocks_to_chunks(doc_id, parsed.blocks)
        for chunk in chunks:
            chunk.score = score_chunk(chunk.text)
        insert_chunks(conn, chunks)

        update_document_status(conn, doc_id, "extracting")
        _extract_all(conn, llm, doc_id, chunks, call_budget)

        budget = explanation_budget if explanation_budget is not None else get_settings().explanation_budget_per_call
        reconcile_all(conn, llm, explanation_budget=budget)
        update_document_status(conn, doc_id, "ready")
    except Exception as exc:
        update_document_status(conn, doc_id, "failed")
        insert_failures(conn, doc_id, [{"reason": f"pipeline_error: {exc}", "kind": "pipeline"}])
    finally:
        conn.close()


def ingest_local_file(conn, llm: LLMClient, file_path: str, filename: str, call_budget: int) -> str:
    """Synchronous, single-connection variant of the pipeline for the seed script.

    Like run_ingest_pipeline, extraction failures (e.g. a cold cache with no API
    key causing 100% of a document's budgeted chunks to fail) are caught and
    degrade the document to "failed" with a pipeline_error logged, rather than
    propagating out of this function and crashing the caller (e.g. the seed
    script partway through a multi-document batch).
    """
    with open(file_path, "rb") as f:
        content = f.read()
    sha256 = hashlib.sha256(content).hexdigest()
    existing = get_document_by_sha256(conn, sha256)
    if existing:
        doc_id = existing.id
    else:
        doc_id = str(uuid.uuid4())
        insert_document(
            conn, doc_id, filename, sha256, page_count=0,
            uploaded_at=datetime.now(timezone.utc).isoformat(), status="pending",
        )

    update_document_status(conn, doc_id, "parsing")
    parsed = parse_pdf(file_path)
    set_document_metadata(conn, doc_id, parsed.page_count, parsed.doc_date)

    chunks = blocks_to_chunks(doc_id, parsed.blocks)
    for chunk in chunks:
        chunk.score = score_chunk(chunk.text)
    insert_chunks(conn, chunks)

    update_document_status(conn, doc_id, "extracting")
    try:
        _extract_all(conn, llm, doc_id, chunks, call_budget)
    except Exception as exc:
        update_document_status(conn, doc_id, "failed")
        insert_failures(conn, doc_id, [{"reason": f"pipeline_error: {exc}", "kind": "pipeline"}])
        return doc_id

    update_document_status(conn, doc_id, "ready")
    return doc_id


def reconcile_all(conn, llm: LLMClient, explanation_budget: int | None = None) -> None:
    """Group all claims across all documents, reconcile new pairs, and explain the interesting ones.

    Idempotent: pairs already present in the relations table (by claim id pair)
    are skipped on subsequent calls, and insert_relations uses INSERT OR IGNORE
    against the (claim_a_id, claim_b_id) UNIQUE constraint as a second safety net.

    Relations are inserted FIRST, with rule_verdict == final_verdict and no
    explanation (the default state reconcile() already produces), before any
    LLM explanation call runs. This means an interrupt mid-explanation-loop
    only loses explanations not yet attempted -- every relation computed this
    call is already durably persisted.

    GATE 3 (RECONCILED_BY_CONTEXT for any envelope-differing pair) is the
    COMMON case, not rare, so explaining every CONTRADICTS/RECONCILED_BY_CONTEXT
    relation with a live LLM call in one invocation has no natural cap. Only
    up to `explanation_budget` "interesting" relations (per reason/explain.py's
    INTERESTING_VERDICTS) are sent to explain_relation() per call, in the order
    they were computed; relations beyond the budget stay in their inserted
    default state (final_verdict == rule_verdict, explanation is None) --
    persisted, just not explained yet. A later reconcile_all() call does not
    revisit them (only genuinely new pairs are considered), matching the
    "process in order, don't over-engineer prioritization" scope for this fix.
    """
    if explanation_budget is None:
        explanation_budget = get_settings().explanation_budget_per_call

    all_claims = get_all_claims_with_vintage(conn)
    groups = group_claims(all_claims)
    seen_pairs = get_relation_pairs_seen(conn)
    pending: list[tuple] = []

    for claims_in_group in groups.values():
        for i in range(len(claims_in_group)):
            for j in range(i + 1, len(claims_in_group)):
                a, b = claims_in_group[i], claims_in_group[j]
                if (a.id, b.id) in seen_pairs:
                    continue
                relation = reconcile(a, b)
                relation.id = str(uuid.uuid4())
                pending.append((relation, a, b))
                seen_pairs.add((a.id, b.id))

    insert_relations(conn, [r for r, _, _ in pending])

    to_explain = [(r, a, b) for r, a, b in pending if r.rule_verdict in INTERESTING_VERDICTS]
    for relation, a, b in to_explain[:explanation_budget]:
        explain_relation(llm, relation, a, b)
        update_relation_explanation(
            conn, relation.id, relation.final_verdict, relation.explanation,
            relation.llm_overrode, relation.override_reason,
        )
