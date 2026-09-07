from app.models import Claim
from reason.align import slugify, group_claims


def _claim(claim_id, subject_key, measure_key):
    return Claim(
        id=claim_id, doc_id="d1", chunk_id="c1", subject=subject_key, subject_key=subject_key,
        measure=measure_key, measure_key=measure_key, value_type="numeric", value_num=1.0,
        value_text=None, unit_raw=None, unit_dim=None, unit_scale=1.0, period_start=None,
        period_end=None, period_label=None, basis={}, evidence_page=1, evidence_quote="x",
        evidence_char_start=0, confidence=1.0, extractor_version="v1",
    )


def test_slugify_normalizes_punctuation_and_case():
    assert slugify("Delhivery Limited") == "delhivery_limited"
    assert slugify("Real GDP Growth (Rate)") == "real_gdp_growth_rate"
    assert slugify("  extra   spaces ") == "extra_spaces"


def test_slugify_empty_falls_back():
    assert slugify("") == "unknown"
    assert slugify("!!!") == "unknown"


def test_group_claims_by_subject_and_measure():
    claims = [
        _claim("1", "rbi", "real_gdp_growth"),
        _claim("2", "imf", "real_gdp_growth"),
        _claim("3", "rbi", "current_account_deficit"),
    ]
    groups = group_claims(claims)
    assert set(groups.keys()) == {("rbi", "real_gdp_growth"), ("imf", "real_gdp_growth"), ("rbi", "current_account_deficit")}
    assert len(groups[("rbi", "real_gdp_growth")]) == 1
