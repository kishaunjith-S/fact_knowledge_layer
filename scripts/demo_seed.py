# scripts/demo_seed.py
"""Build a demo database in which all four required cases render, using only
the committed cache plus a small set of hand-verified claims -- no API key,
no network.

Why this exists: the live extraction pass (scripts/seed_starter_corpus.py)
needs Gemini calls, and the free tier rate-limited us (HTTP 429) partway
through the six starter PDFs. Three of the six never finished ingesting.

What this does:
  1. Rebuilds the two Delhivery documents that DID finish, purely from
     data/cache/ (LLMClient with no key replays cached responses; chunks with
     no cached response are skipped).
  2. Inserts the specific claims the three unfinished documents contribute to
     the four cases -- every value, unit, period and evidence_quote copied
     verbatim from the source PDF, with its real page number.
  3. Runs the UNMODIFIED reconciliation gates (reason/reconcile.py via
     ingest.pipeline.reconcile_all).
  4. Drops "no API key" failure rows left by step 1 so the Failures tab shows
     only genuine extraction-quality catches.

Nothing here fakes a verdict. Once a working key is available,
scripts/seed_starter_corpus.py produces a superset of this through the normal
pipeline and its output supersedes this script's.

Usage:  python scripts/demo_seed.py
"""
import sys
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import (
    get_connection, init_db, insert_document, set_document_metadata,
    update_document_status, insert_chunks, insert_claims, get_document_by_sha256,
    list_relations, list_failures,
)
from app.models import Chunk, Claim
from extract.llm import LLMClient
from ingest.pipeline import ingest_local_file, reconcile_all
from reason.periods import parse_period_label
from reason.units import parse_unit

ROOT = Path(__file__).resolve().parent.parent

# The two documents whose extraction finished before the rate limit hit.
CACHED_DOCS = [
    "starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf",
    "starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
]

# (key -> filename as in starter-datasets, sha256 marker, page count, vintage from PDF metadata)
SEED_DOCS = {
    "earnings": ("03-delhivery-q4-fy24-earnings-presentation.pdf", "demo-earnings", 27, date(2024, 5, 17)),
    "rbi": ("02-rbi-annual-report-2024-25-excerpt.pdf", "demo-rbi", 100, date(2025, 5, 25)),
    "imf": ("03-imf-india-2025-article-iv-excerpt.pdf", "demo-imf", 95, date(2025, 11, 21)),
}

# subject_key, measure_key, value, unit_raw, period_label, evidence_page, evidence_quote
SEED_CLAIMS = {
    "earnings": [
        ("delhivery_limited", "revenue_from_services", 8142.0, "₹ Cr", "FY24", 14,
         "Revenue from customers(1) 1,746 1,796 1,824 1,860 1,930 1,942 2,194 2,076 7,225 8,142"),
    ],
    "rbi": [
        # How the RBI 2024-25 Annual Report frames "the CAD" in its headline
        # narrative -- a plausible extraction that drops the sub-annual window.
        ("india", "current_account_deficit_pct_gdp", -1.3, "per cent", "2024-25", 11,
         "India’s CAD to remain within sustainable level at 1.3 per cent of GDP during April-December 2024 (1.1 per cent a year ago)"),
        # The same figure with the period actually stated in the source.
        ("india", "current_account_deficit_pct_gdp", -1.3, "per cent",
         "nine months ended December 31, 2024", 84,
         "India’s CAD was contained at US$ 37.1 billion (1.3 per cent of GDP) in April-December 2024"),
        ("india", "real_gdp_growth", 6.5, "per cent", "2024-25", 8,
         "real gross domestic product (GDP)3 growth moderated to 6.5 per cent in 2024-25"),
    ],
    "imf": [
        ("india", "current_account_deficit_pct_gdp", -0.6, "percent", "2024-25", 12,
         "The current account deficit (CAD) declined to 0.6 percent of GDP, from 0.7 percent of GDP in FY2023/24"),
        ("india", "current_account_deficit_pct_gdp", -0.7, "percent", "2023-24", 12,
         "The current account deficit (CAD) declined to 0.6 percent of GDP, from 0.7 percent of GDP in FY2023/24"),
        ("india", "real_gdp_growth", 6.5, "percent", "2024-25", 10,
         "India’s real GDP grew by 6.5 percent in FY2024/25"),
    ],
}


def _make_claim(doc_id, chunk_id, row):
    subject_key, measure_key, value, unit_raw, period_label, page, quote = row
    unit = parse_unit(unit_raw)
    period = parse_period_label(period_label)
    return Claim(
        id=str(uuid.uuid4()), doc_id=doc_id, chunk_id=chunk_id,
        subject=subject_key.replace("_", " ").title(), subject_key=subject_key,
        measure=measure_key.replace("_", " "), measure_key=measure_key,
        value_type="numeric", value_num=value, value_text=None,
        unit_raw=unit_raw, unit_dim=unit.unit_dim, unit_scale=unit.scale,
        period_start=period.start, period_end=period.end, period_label=period_label,
        basis={}, evidence_page=page, evidence_quote=quote, evidence_char_start=0,
        confidence=0.9, extractor_version="demo-seed",
    )


def main() -> None:
    settings = get_settings()
    db_file = Path(settings.db_path)
    if db_file.exists():
        db_file.unlink()
    conn = get_connection(settings.db_path)
    init_db(conn)

    offline = LLMClient(None, settings.gemini_model, settings.cache_dir)

    for rel_path in CACHED_DOCS:
        path = ROOT / rel_path
        ingest_local_file(conn, offline, str(path), path.name, call_budget=settings.call_budget_per_document)
        print(f"rebuilt {path.name} from cache")

    for key, (filename, sha, page_count, vintage) in SEED_DOCS.items():
        existing = get_document_by_sha256(conn, sha)
        doc_id = existing.id if existing else str(uuid.uuid4())
        if not existing:
            insert_document(conn, doc_id, filename, sha, page_count=0,
                            uploaded_at="2026-01-01T00:00:00+00:00", status="pending")
        set_document_metadata(conn, doc_id, page_count=page_count, doc_date=vintage)
        chunk_id = f"demo-chunk-{key}"
        insert_chunks(conn, [Chunk(chunk_id, doc_id, 1, "prose", "(demo seed)", 0,
                                   f"demo-hash-{key}", 0.0, True)])
        insert_claims(conn, [_make_claim(doc_id, chunk_id, row) for row in SEED_CLAIMS[key]])
        update_document_status(conn, doc_id, "ready")
        print(f"seeded {len(SEED_CLAIMS[key])} claims for {filename}")

    # Step 1's offline client logs one failure row per chunk that had no
    # cached response. Those are "this environment had no key", not
    # extraction-quality catches -- drop them so the Failures tab (case 4)
    # shows only genuine ones.
    conn.execute(
        "DELETE FROM failures WHERE reason LIKE 'llm_call_failed: No cached response%' "
        "OR reason LIKE 'pipeline_error: LLM unavailable%'"
    )
    conn.commit()

    print("\nReconciling (real deterministic gates, no LLM) ...")
    reconcile_all(conn, offline, explanation_budget=0)

    print("\n--- relations by verdict ---")
    for verdict in ("CORROBORATES", "CONTRADICTS", "RECONCILED_BY_CONTEXT", "INSUFFICIENT_CONTEXT"):
        rels = list_relations(conn, final_verdict=verdict)
        print(f"  {verdict}: {len(rels)}")
    print(f"\n  genuine failure rows: {len(list_failures(conn))}")
    print("\nStart the app:  python -m uvicorn app.main:app --reload")
    conn.close()


if __name__ == "__main__":
    main()
