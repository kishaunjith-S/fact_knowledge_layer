# scripts/seed_starter_corpus.py
"""Ingest the six starter PDFs and verify the four required cases surface
with the expected rule_fired values. Requires GEMINI_API_KEY on its first
run (to populate data/cache/); reruns replay entirely from that committed
cache with no key needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.db import get_connection, init_db, list_relations, list_failures
from extract.llm import LLMClient
from ingest.pipeline import ingest_local_file, reconcile_all

STARTER_FILES = [
    "starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf",
    "starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
    "starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
    "starter-datasets/india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf",
    "starter-datasets/india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf",
    "starter-datasets/india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf",
]

# One rule per required case (spec Section 11). Presence of at least one
# relation with each rule_fired is how this script confirms the four cases
# actually surfaced from real ingestion, not just from the unit tests.
EXPECTED_RULES_BY_CASE = {
    "case 1 (corroborated)": ["exact_envelope_match", "unit_normalized_match"],
    "case 2 (genuine contradiction)": ["large_gap_despite_vintage_gap"],
    "case 3 (explained by context)": ["envelope_difference_explains", "later_vintage_revision"],
}


def main() -> None:
    settings = get_settings()
    conn = get_connection(settings.db_path)
    init_db(conn)
    llm = LLMClient(settings.gemini_api_key, settings.gemini_model, settings.cache_dir)

    root = Path(__file__).resolve().parent.parent
    for relative_path in STARTER_FILES:
        path = root / relative_path
        print(f"Ingesting {relative_path} ...")
        ingest_local_file(conn, llm, str(path), path.name, call_budget=settings.call_budget_per_document)

    print("Reconciling across the full corpus ...")
    reconcile_all(conn, llm, explanation_budget=settings.explanation_budget_per_call)

    print("\n--- Verification ---")
    rules_seen = {r.rule_fired for r in list_relations(conn)}
    for case_name, candidate_rules in EXPECTED_RULES_BY_CASE.items():
        found = rules_seen & set(candidate_rules)
        status = "FOUND" if found else "MISSING"
        print(f"  [{status}] {case_name}: {found or candidate_rules}")

    failure_count = len(list_failures(conn))
    print(f"  [{'FOUND' if failure_count else 'MISSING'}] case 4 (extraction/reasoning failures): {failure_count} entries")

    conn.close()


if __name__ == "__main__":
    main()
