from app.models import Claim, Relation
from reason.explain import relation_content_hash, explain_relation


def _claim(claim_id, evidence_quote="quote text", subject_key="india", measure_key="cad_pct_gdp"):
    return Claim(
        id=claim_id, doc_id="d1", chunk_id="c1", subject="India", subject_key=subject_key,
        measure="CAD", measure_key=measure_key, value_type="numeric", value_num=-1.3,
        value_text=None, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
        period_start=None, period_end=None, period_label="2024-25", basis={},
        evidence_page=1, evidence_quote=evidence_quote, evidence_char_start=0,
        confidence=0.9, extractor_version="v1",
    )


def _relation(rule_verdict="CONTRADICTS", rule_fired="large_gap_despite_vintage_gap"):
    return Relation(
        id="r1", claim_a_id="a1", claim_b_id="b1", rule_verdict=rule_verdict, final_verdict=rule_verdict,
        axis="vintage", rule_fired=rule_fired, delta_pct=0.54,
    )


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete_json(self, prompt, cache_key):
        self.calls += 1
        return self.response


def test_relation_content_hash_stable_across_different_claim_ids():
    a1 = _claim("uuid-run-1-a")
    b1 = _claim("uuid-run-1-b", subject_key="india", measure_key="cad_pct_gdp", evidence_quote="quote text b")
    a2 = _claim("uuid-run-2-a")
    b2 = _claim("uuid-run-2-b", subject_key="india", measure_key="cad_pct_gdp", evidence_quote="quote text b")
    assert relation_content_hash(a1, b1, "large_gap_despite_vintage_gap") == \
        relation_content_hash(a2, b2, "large_gap_despite_vintage_gap")


def test_relation_content_hash_differs_for_different_evidence():
    a = _claim("a1")
    b1 = _claim("b1", evidence_quote="quote one")
    b2 = _claim("b2", evidence_quote="quote two")
    assert relation_content_hash(a, b1, "rule") != relation_content_hash(a, b2, "rule")


def test_uninteresting_verdict_skips_llm_entirely():
    llm = FakeLLM({"explanation": "should not be used", "override": None, "override_reason": None})
    relation = _relation(rule_verdict="CORROBORATES", rule_fired="exact_envelope_match")
    result = explain_relation(llm, relation, _claim("a1"), _claim("b1"))
    assert llm.calls == 0
    assert result.final_verdict == "CORROBORATES"
    assert result.llm_overrode is False


def test_contradicts_calls_llm_and_keeps_verdict_without_override():
    llm = FakeLLM({"explanation": "Both measure the same thing but disagree.", "override": None, "override_reason": None})
    relation = _relation(rule_verdict="CONTRADICTS", rule_fired="large_gap_despite_vintage_gap")
    result = explain_relation(llm, relation, _claim("a1"), _claim("b1"))
    assert llm.calls == 1
    assert result.final_verdict == "CONTRADICTS"
    assert result.explanation == "Both measure the same thing but disagree."
    assert result.llm_overrode is False


def test_llm_override_is_applied_and_logged():
    llm = FakeLLM({
        "explanation": "The RBI figure is explicitly provisional and likely to be revised.",
        "override": "RECONCILED_BY_CONTEXT", "override_reason": "RBI figure marked (P) provisional",
    })
    relation = _relation(rule_verdict="CONTRADICTS", rule_fired="large_gap_despite_vintage_gap")
    result = explain_relation(llm, relation, _claim("a1"), _claim("b1"))
    assert result.final_verdict == "RECONCILED_BY_CONTEXT"
    assert result.llm_overrode is True
    assert result.override_reason == "RBI figure marked (P) provisional"
    assert result.rule_verdict == "CONTRADICTS"  # unmodified


def test_override_equal_to_rule_verdict_is_not_treated_as_override():
    llm = FakeLLM({"explanation": "Confirmed.", "override": "CONTRADICTS", "override_reason": "still contradicts"})
    relation = _relation(rule_verdict="CONTRADICTS", rule_fired="large_gap_despite_vintage_gap")
    result = explain_relation(llm, relation, _claim("a1"), _claim("b1"))
    assert result.llm_overrode is False
    assert result.final_verdict == "CONTRADICTS"


def test_llm_failure_falls_back_to_rule_verdict():
    class RaisingLLM:
        def complete_json(self, prompt, cache_key):
            raise RuntimeError("network error")

    relation = _relation(rule_verdict="RECONCILED_BY_CONTEXT", rule_fired="envelope_difference_explains")
    result = explain_relation(RaisingLLM(), relation, _claim("a1"), _claim("b1"))
    assert result.final_verdict == "RECONCILED_BY_CONTEXT"
    assert result.explanation is None
    assert result.llm_overrode is False
