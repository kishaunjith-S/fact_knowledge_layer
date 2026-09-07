# extract/claims.py
import re
import uuid

from app.models import Claim
from extract.llm import LLMClient, cache_key_for
from extract.prompts import EXTRACTION_PROMPT_TEMPLATE, EXTRACTION_PROMPT_VERSION, build_registry_block
from extract.registry import top_measures, register_measure
from reason.align import slugify
from reason.periods import parse_period_label
from reason.units import parse_unit

REQUIRED_FIELDS = ("subject", "measure", "value_type")

_WS_RUN = re.compile(r"\s+")


def _locate_evidence(chunk_text: str, quote: str):
    """Find the model's evidence quote in the chunk, tolerant of the newlines
    pymupdf inserts at every PDF line wrap.

    A quote copied verbatim from a sentence that wrapped mid-line arrives from
    the model with the wrap collapsed to a space, so an exact substring check
    rejects a perfectly grounded citation. Match with every run of whitespace
    treated as equivalent, and return the span in the ORIGINAL text so
    evidence offsets stay exact. Returns an re.Match or None.
    """
    collapsed = _WS_RUN.sub(" ", quote).strip()
    if not collapsed:
        return None
    pattern = r"\s+".join(re.escape(token) for token in collapsed.split(" "))
    return re.search(pattern, chunk_text)


def extract_claims_for_chunk(conn, llm: LLMClient, chunk) -> tuple[list[Claim], list[dict]]:
    registry_block = build_registry_block(top_measures(conn))
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(registry_block=registry_block, chunk_text=chunk.text)
    cache_key = cache_key_for(chunk.content_hash, EXTRACTION_PROMPT_VERSION)

    try:
        raw_items = llm.complete_json(prompt, cache_key)
    except Exception as exc:
        return [], [{"chunk_id": chunk.id, "reason": f"llm_call_failed: {exc}"}]

    if not isinstance(raw_items, list):
        return [], [{"chunk_id": chunk.id, "reason": "response_not_a_list", "raw": {"response": str(raw_items)[:500]}}]

    claims: list[Claim] = []
    failures: list[dict] = []
    for item in raw_items:
        try:
            result = _build_claim(conn, chunk, item)
        except Exception as exc:
            result = {"chunk_id": chunk.id, "reason": f"malformed_item: {exc}", "raw": item}
        if isinstance(result, Claim):
            claims.append(result)
        else:
            failures.append(result)
    return claims, failures


def _build_claim(conn, chunk, item: dict):
    if not isinstance(item, dict):
        return {"chunk_id": chunk.id, "reason": "item_not_an_object", "raw": {"item": str(item)[:500]}}

    for field_name in REQUIRED_FIELDS:
        if not item.get(field_name):
            return {"chunk_id": chunk.id, "reason": f"missing_field:{field_name}", "raw": item}

    match = _locate_evidence(chunk.text, item.get("evidence_quote") or "")
    if match is None:
        return {"chunk_id": chunk.id, "reason": "evidence_quote_not_found_in_chunk", "raw": item}
    quote = match.group(0)

    value_type = item["value_type"]
    value_num = item.get("value_num")
    if value_type == "numeric" and not isinstance(value_num, (int, float)):
        return {"chunk_id": chunk.id, "reason": "numeric_claim_missing_value_num", "raw": item}

    subject_key = slugify(item.get("subject_key") or item["subject"])
    measure_key = slugify(item.get("measure_key") or item["measure"])
    unit_info = parse_unit(item.get("unit_raw"))
    period_info = parse_period_label(item.get("period_label"))

    # Compute every field value the Claim(...) constructor below needs into
    # local variables FIRST -- in particular confidence's float(...) conversion,
    # which can still raise on a non-numeric value from the model. All of this
    # must happen BEFORE register_measure() runs, so a rejected item (one that
    # fails to become a Claim) never leaves a phantom measure_key committed to
    # measure_registry. A phantom key would otherwise get fed back into every
    # subsequent extraction prompt via top_measures() -- the exact slug-drift
    # problem the registry exists to prevent.
    claim_id = str(uuid.uuid4())
    value_num_final = float(value_num) if value_num is not None else None
    confidence = float(item.get("confidence", 0.5))
    evidence_char_start = match.start()
    basis = item.get("basis") or {}

    # register_measure's alias argument tracks distinct surface forms a
    # measure has been called. Passing item["measure"] (the label) as both
    # label and alias made alias-tracking inert -- an alias can never equal
    # the label it's an alias of. Pass the model's raw, pre-slug measure_key
    # hint instead (which may differ from the final slugified measure_key),
    # or None if the model didn't supply one.
    measure_key_alias = item.get("measure_key")
    register_measure(conn, measure_key, item["measure"], measure_key_alias, unit_info.unit_dim)

    return Claim(
        id=claim_id, doc_id=chunk.doc_id, chunk_id=chunk.id,
        subject=item["subject"], subject_key=subject_key,
        measure=item["measure"], measure_key=measure_key,
        value_type=value_type,
        value_num=value_num_final,
        value_text=item.get("value_text"),
        unit_raw=item.get("unit_raw"), unit_dim=unit_info.unit_dim, unit_scale=unit_info.scale,
        period_start=period_info.start, period_end=period_info.end, period_label=item.get("period_label"),
        basis=basis,
        evidence_page=chunk.page_no, evidence_quote=quote, evidence_char_start=evidence_char_start,
        confidence=confidence, extractor_version=EXTRACTION_PROMPT_VERSION,
    )
