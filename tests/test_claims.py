from app.db import get_connection, init_db
from app.models import Chunk
from extract.claims import extract_claims_for_chunk


def make_conn():
    conn = get_connection(":memory:")
    init_db(conn)
    return conn


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete_json(self, prompt, cache_key):
        self.calls += 1
        return self.response


def _chunk(text="Revenue from operations stood at Rs 100 crore in FY2024."):
    return Chunk(id="c1", doc_id="d1", page_no=5, kind="prose", text=text,
                 char_start=0, content_hash="hash1", score=1.0, extracted=False)


def test_valid_item_builds_claim_with_parsed_unit_and_period():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": "Delhivery", "subject_key": "delhivery_limited",
        "measure": "Revenue from Operations", "measure_key": "revenue_from_operations",
        "value_type": "numeric", "value_num": 100, "value_text": None,
        "unit_raw": "Rs crore", "period_label": "FY2024", "basis": {"consolidated": True},
        "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
        "confidence": 0.9,
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert failures == []
    assert len(claims) == 1
    claim = claims[0]
    assert claim.subject_key == "delhivery_limited"
    assert claim.measure_key == "revenue_from_operations"
    assert claim.unit_dim == "currency_inr"
    assert claim.unit_scale == 1e7
    assert claim.period_label == "FY2024"
    assert claim.period_start is not None and claim.period_end is not None
    assert claim.evidence_page == 5
    assert claim.doc_id == "d1"
    assert claim.chunk_id == "c1"


def test_evidence_quote_matches_across_pdf_linewrap_newlines():
    # pymupdf emits a newline at every PDF line wrap, so a sentence's text
    # arrives with newlines in the middle of it. The model copies the quote
    # with those wraps collapsed to spaces (as any reader would). An exact
    # substring check then rejects a perfectly grounded citation -- this is
    # how the prospectus 9-month "total income 49,114.06" evidence was lost.
    conn = make_conn()
    chunk = _chunk(
        "to ₹29,886.29 million in Fiscal 2020 and to ₹38,382.91 million\n"
        "in Fiscal 2021, and from ₹28,065.29 million for the nine months\n"
        "period ended December 31, 2021."
    )
    llm = FakeLLM([{
        "subject": "Our Company", "measure": "Total income", "value_type": "numeric",
        "value_num": 38382.91, "unit_raw": "million", "period_label": "Fiscal 2021",
        "evidence_quote": "to ₹38,382.91 million in Fiscal 2021",
        "confidence": 0.9,
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert failures == []
    assert len(claims) == 1
    claim = claims[0]
    # the stored quote is the verbatim source span (newline and all) and its
    # offset indexes back into the original chunk text exactly
    assert claim.evidence_quote in chunk.text
    span = chunk.text[claim.evidence_char_start:claim.evidence_char_start + len(claim.evidence_quote)]
    assert span == claim.evidence_quote


def test_hallucinated_evidence_quote_is_rejected():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": "Delhivery", "measure": "Revenue", "value_type": "numeric", "value_num": 999,
        "evidence_quote": "this text does not appear anywhere in the chunk",
        "confidence": 0.5,
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert claims == []
    assert len(failures) == 1
    assert failures[0]["reason"] == "evidence_quote_not_found_in_chunk"


def test_missing_required_field_is_rejected():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{"measure": "Revenue", "value_type": "numeric", "evidence_quote": "Rs 100 crore"}])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert claims == []
    assert failures[0]["reason"] == "missing_field:subject"


def test_numeric_claim_missing_value_num_is_rejected():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": "Delhivery", "measure": "Revenue", "value_type": "numeric", "value_num": None,
        "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert claims == []
    assert failures[0]["reason"] == "numeric_claim_missing_value_num"


def test_missing_measure_key_falls_back_to_slugified_measure():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": "Delhivery", "subject_key": "delhivery_limited", "measure": "Revenue From Operations",
        "value_type": "numeric", "value_num": 100,
        "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert failures == []
    assert claims[0].measure_key == "revenue_from_operations"


def test_response_not_a_list_is_a_failure():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM({"not": "a list"})
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert claims == []
    assert failures[0]["reason"] == "response_not_a_list"


def test_llm_call_failure_is_captured_not_raised():
    conn = make_conn()
    chunk = _chunk()

    class RaisingLLM:
        def complete_json(self, prompt, cache_key):
            raise RuntimeError("network error")

    claims, failures = extract_claims_for_chunk(conn, RaisingLLM(), chunk)
    assert claims == []
    assert "llm_call_failed" in failures[0]["reason"]


def test_malformed_item_type_is_rejected_not_raised():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": 12345,  # non-string subject: slugify() would raise AttributeError
        "measure": "Revenue", "value_type": "numeric", "value_num": 100,
        "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert claims == []
    assert len(failures) == 1
    assert failures[0]["reason"].startswith("malformed_item:")
    assert failures[0]["raw"]["subject"] == 12345


def test_non_string_evidence_quote_is_rejected_not_raised():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": "Delhivery", "measure": "Revenue", "value_type": "numeric", "value_num": 100,
        "evidence_quote": 42,  # non-string quote: "quote not in chunk.text" would raise TypeError
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert claims == []
    assert len(failures) == 1
    assert failures[0]["reason"].startswith("malformed_item:")


def test_non_numeric_confidence_is_rejected_not_raised():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": "Delhivery", "measure": "Revenue", "value_type": "numeric", "value_num": 100,
        "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
        "confidence": "not-a-number",  # float(...) would raise ValueError
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert claims == []
    assert len(failures) == 1
    assert failures[0]["reason"].startswith("malformed_item:")


def test_malformed_item_does_not_prevent_other_items_in_batch_from_succeeding():
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([
        {
            "subject": 12345,  # malformed: will raise inside _build_claim
            "measure": "Revenue", "value_type": "numeric", "value_num": 100,
            "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
        },
        {
            "subject": "Delhivery", "subject_key": "delhivery_limited",
            "measure": "Revenue from Operations", "measure_key": "revenue_from_operations",
            "value_type": "numeric", "value_num": 100, "unit_raw": "Rs crore", "period_label": "FY2024",
            "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
        },
    ])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert len(failures) == 1
    assert failures[0]["reason"].startswith("malformed_item:")
    assert len(claims) == 1
    assert claims[0].measure_key == "revenue_from_operations"


def test_non_numeric_confidence_does_not_leave_phantom_registry_entry():
    # Regression test: register_measure() used to be called BEFORE Claim(...)
    # construction, but Claim(...)'s confidence=float(item.get("confidence",
    # 0.5)) can still raise AFTER the registry write already committed --
    # leaving a phantom measure_key in measure_registry for an item that was
    # ultimately rejected. That phantom key would then be fed back into every
    # subsequent extraction prompt via top_measures(), reintroducing the
    # slug-drift problem the registry exists to prevent.
    from extract.registry import top_measures
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": "Delhivery", "subject_key": "delhivery_limited",
        "measure": "Revenue from Operations", "measure_key": "revenue_from_operations",
        "value_type": "numeric", "value_num": 100,
        "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
        "confidence": "not-a-number",  # float(...) raises ValueError
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert claims == []
    assert len(failures) == 1
    assert failures[0]["reason"].startswith("malformed_item:")

    measures = top_measures(conn)
    assert not any(m["measure_key"] == "revenue_from_operations" for m in measures)


def test_register_measure_alias_uses_llm_measure_key_hint_not_label():
    # Regression test: register_measure(conn, measure_key, item["measure"],
    # item.get("measure"), ...) passed item["measure"] as BOTH label and
    # alias, so the alias list could never contain a surface form distinct
    # from the label -- alias-tracking was inert. The alias should instead be
    # the model's raw, pre-slug measure_key candidate (which may differ from
    # both the label and the final slugified measure_key).
    from extract.registry import top_measures
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": "Delhivery", "subject_key": "delhivery_limited",
        "measure": "Revenue from Operations", "measure_key": "Revenue Ops Hint",
        "value_type": "numeric", "value_num": 100, "unit_raw": "Rs crore", "period_label": "FY2024",
        "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
    }])
    claims, failures = extract_claims_for_chunk(conn, llm, chunk)
    assert failures == []
    assert claims[0].measure_key == "revenue_ops_hint"

    measures = top_measures(conn)
    entry = next(m for m in measures if m["measure_key"] == "revenue_ops_hint")
    assert entry["label"] == "Revenue from Operations"
    assert "Revenue Ops Hint" in entry["aliases"]


def test_valid_claim_registers_measure_in_registry():
    from extract.registry import top_measures
    conn = make_conn()
    chunk = _chunk()
    llm = FakeLLM([{
        "subject": "Delhivery", "subject_key": "delhivery_limited",
        "measure": "Revenue from Operations", "measure_key": "revenue_from_operations",
        "value_type": "numeric", "value_num": 100, "unit_raw": "Rs crore", "period_label": "FY2024",
        "evidence_quote": "Revenue from operations stood at Rs 100 crore in FY2024.",
    }])
    extract_claims_for_chunk(conn, llm, chunk)
    measures = top_measures(conn)
    assert any(m["measure_key"] == "revenue_from_operations" for m in measures)
