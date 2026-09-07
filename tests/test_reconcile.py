# tests/test_reconcile.py
from datetime import date

from app.models import Claim
from reason.reconcile import (
    basis_differs, subject_granularity_differs, vintage_gap_days, reconcile,
    MAGNITUDE_CAP, VINTAGE_GAP_DAYS,
)


def _claim(**overrides):
    base = dict(
        id="c", doc_id="d", chunk_id="ch", subject="Delhivery", subject_key="delhivery_limited",
        measure="Revenue", measure_key="revenue_from_operations", value_type="numeric",
        value_num=100.0, value_text=None, unit_raw="₹ crore", unit_dim="currency_inr", unit_scale=1e7,
        period_start=date(2023, 4, 1), period_end=date(2024, 3, 31), period_label="FY2024",
        basis={}, evidence_page=1, evidence_quote="quote", evidence_char_start=0,
        confidence=0.9, extractor_version="v1", doc_vintage=None,
    )
    base.update(overrides)
    return Claim(**base)


# ---- basis_differs ----

def test_basis_differs_shared_key_mismatch():
    differs, key = basis_differs({"consolidated": True}, {"consolidated": False})
    assert differs is True and key == "consolidated"


def test_basis_differs_unstated_key_is_not_a_difference():
    differs, key = basis_differs({"consolidated": True}, {})
    assert differs is False and key is None


def test_basis_differs_restated_asserted_on_one_side_counts():
    differs, key = basis_differs({"restated": True}, {})
    assert differs is True and key == "restated"


def test_basis_differs_projected_asserted_on_one_side_counts():
    differs, key = basis_differs({}, {"projected": True})
    assert differs is True and key == "projected"


def test_basis_differs_no_shared_keys_and_no_restated_projected():
    differs, key = basis_differs({"consolidated": True}, {"segment": "express"})
    assert differs is False and key is None


# ---- subject_granularity_differs ----

def test_subject_granularity_prefix_match():
    assert subject_granularity_differs("delhivery_limited", "delhivery_limited_group") is True
    assert subject_granularity_differs("delhivery_limited_group", "delhivery_limited") is True


def test_subject_granularity_unrelated_keys():
    assert subject_granularity_differs("delhivery_limited", "rbi") is False


def test_subject_granularity_identical_keys():
    assert subject_granularity_differs("rbi", "rbi") is False


# ---- vintage_gap_days ----

def test_vintage_gap_days_computed():
    assert vintage_gap_days(date(2025, 5, 25), date(2025, 11, 21)) == 180


def test_vintage_gap_days_none_when_either_missing():
    assert vintage_gap_days(None, date(2025, 1, 1)) is None


# ---- GATE 0: incommensurable ----

def test_gate0_incommensurable_units():
    a = _claim(unit_dim="percent", unit_scale=1.0, value_num=5.0)
    b = _claim(unit_dim="currency_inr", unit_scale=1e7, value_num=5.0)
    relation = reconcile(a, b)
    assert relation.rule_verdict == "INSUFFICIENT_CONTEXT"
    assert relation.axis == "unit"
    assert relation.rule_fired == "incommensurable_units"


def test_gate0_incommensurable_value_types():
    a = _claim(value_type="numeric", value_num=5.0)
    b = _claim(value_type="text", value_num=None, value_text="active")
    relation = reconcile(a, b)
    assert relation.rule_verdict == "INSUFFICIENT_CONTEXT"
    assert relation.rule_fired == "incommensurable_value_types"


# ---- GATE 1: incomplete envelope ----

def test_gate1_missing_period_end():
    a = _claim(period_end=None)
    b = _claim()
    relation = reconcile(a, b)
    assert relation.rule_verdict == "INSUFFICIENT_CONTEXT"
    assert relation.rule_fired == "incomplete_envelope"


def test_gate1_unclear_period_relationship_missing_start_only():
    a = _claim(period_start=None, period_end=date(2024, 3, 31))
    b = _claim(period_start=date(2023, 4, 1), period_end=date(2024, 3, 31))
    relation = reconcile(a, b)
    assert relation.rule_verdict == "INSUFFICIENT_CONTEXT"
    assert relation.axis == "period"
    assert relation.rule_fired == "unclear_period_relationship"


# ---- GATE 2: identical envelopes ----

def test_gate2_corroborates_exact_match_gdp_growth():
    a = _claim(
        subject_key="rbi", value_num=6.5, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
        doc_vintage=date(2025, 5, 25),
    )
    b = _claim(
        subject_key="rbi", value_num=6.5, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
        doc_vintage=date(2025, 5, 25),
    )
    relation = reconcile(a, b)
    assert relation.rule_verdict == "CORROBORATES"
    assert relation.rule_fired == "exact_envelope_match"


def test_gate2_corroborates_unit_normalized_delhivery_revenue():
    a = _claim(value_num=81415.38, unit_raw="₹ million", unit_dim="currency_inr", unit_scale=1e6,
               doc_vintage=date(2024, 8, 8))
    b = _claim(value_num=8142, unit_raw="₹ crore", unit_dim="currency_inr", unit_scale=1e7,
               doc_vintage=date(2024, 5, 17))
    relation = reconcile(a, b)
    assert relation.rule_verdict == "CORROBORATES"
    assert relation.rule_fired == "unit_normalized_match"


def test_gate2_contradicts_no_vintage_gap():
    a = _claim(value_num=100.0, doc_vintage=date(2024, 8, 8))
    b = _claim(value_num=150.0, doc_vintage=date(2024, 8, 20))  # 12 days apart
    relation = reconcile(a, b)
    assert relation.rule_verdict == "CONTRADICTS"
    assert relation.rule_fired == "same_envelope_value_mismatch"


def test_gate2_reconciled_by_vintage_small_gap_under_magnitude_cap():
    # 0.5pp gap on a ~7-point base is small enough to be a plausible revision
    # but large enough to clear even the widened (rounded-number) tolerance.
    a = _claim(value_num=6.5, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
               doc_vintage=date(2025, 1, 30))
    b = _claim(value_num=7.0, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
               doc_vintage=date(2025, 5, 25))
    relation = reconcile(a, b)
    assert relation.rule_verdict == "RECONCILED_BY_CONTEXT"
    assert relation.axis == "vintage"
    assert relation.rule_fired == "later_vintage_revision"


def test_gate2_corroborates_despite_vintage_gap_within_rounding_tolerance():
    # Documented real finding: Economic Survey's First Advance Estimate (6.4%)
    # vs RBI/IMF's later Provisional Estimate (6.5%) for FY2024-25 GDP growth
    # (spec Section 11 case 3) has a delta of only ~1.5%, which the rounding
    # heuristic classifies as agreement (both sides are 2-significant-figure
    # percentages) before vintage is even considered. This is judged correct,
    # not a bug: two sources essentially agreeing, one merely less precise, is
    # exactly what CORROBORATES should report. It also means this specific
    # pair does not exercise the vintage-explained path -- the CAD pair
    # (case 2, above) is the real demonstration of vintage-gap logic.
    a = _claim(value_num=6.4, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
               doc_vintage=date(2025, 1, 30))
    b = _claim(value_num=6.5, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
               doc_vintage=date(2025, 5, 25))
    relation = reconcile(a, b)
    assert relation.rule_verdict == "CORROBORATES"


def test_gate2_magnitude_cap_forces_contradicts_cad_case():
    # RBI (P) -1.3% vs IMF -0.6%, FY2024-25, vintage gap ~180 days -- spec Section 11 case 2
    rbi = _claim(subject_key="india", measure_key="current_account_deficit_pct_gdp",
                 value_num=-1.3, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
                 basis={"provisional": True}, doc_vintage=date(2025, 5, 25))
    imf = _claim(subject_key="india", measure_key="current_account_deficit_pct_gdp",
                 value_num=-0.6, unit_raw="per cent", unit_dim="percent", unit_scale=1.0,
                 basis={}, doc_vintage=date(2025, 11, 21))
    relation = reconcile(rbi, imf)
    assert relation.rule_verdict == "CONTRADICTS"
    assert relation.axis == "vintage"
    assert relation.rule_fired == "large_gap_despite_vintage_gap"
    assert relation.delta_pct > MAGNITUDE_CAP


def test_gate2_non_numeric_status_change_reconciled_by_vintage():
    active = _claim(value_type="text", value_num=None, value_text="active", unit_raw=None,
                     unit_dim=None, doc_vintage=date(2022, 5, 16))
    resigned = _claim(value_type="text", value_num=None, value_text="resigned", unit_raw=None,
                       unit_dim=None, doc_vintage=date(2024, 8, 8))
    relation = reconcile(active, resigned)
    assert relation.rule_verdict == "RECONCILED_BY_CONTEXT"
    assert relation.axis == "vintage"
    assert relation.rule_fired == "later_vintage_revision"


# ---- GATE 3: envelopes differ ----

def test_gate3_period_axis_prospectus_nine_months_vs_full_year():
    nine_months = _claim(
        value_num=49114.06, unit_raw="₹ million", unit_dim="currency_inr", unit_scale=1e6,
        period_start=date(2021, 4, 1), period_end=date(2021, 12, 31), period_label="9M ended 2021-12-31",
        doc_vintage=date(2022, 5, 16),
    )
    full_year = _claim(
        value_num=6800.0, unit_raw="₹ crore", unit_dim="currency_inr", unit_scale=1e7,
        period_start=date(2021, 4, 1), period_end=date(2022, 3, 31), period_label="FY2022",
        doc_vintage=date(2022, 5, 16),
    )
    relation = reconcile(nine_months, full_year)
    assert relation.rule_verdict == "RECONCILED_BY_CONTEXT"
    assert relation.axis == "period"
    assert relation.rule_fired == "envelope_difference_explains"


def test_gate3_period_axis_coincident_values():
    a = _claim(value_num=100.0, period_start=date(2023, 4, 1), period_end=date(2024, 3, 31))
    b = _claim(value_num=100.0, period_start=date(2024, 4, 1), period_end=date(2025, 3, 31))
    relation = reconcile(a, b)
    assert relation.rule_verdict == "RECONCILED_BY_CONTEXT"
    assert relation.axis == "period"
    assert relation.rule_fired == "coincident_values_differing_envelope"


def test_gate3_scope_axis_basis_differs():
    a = _claim(value_num=100.0, basis={"consolidated": True})
    b = _claim(value_num=200.0, basis={"consolidated": False})
    relation = reconcile(a, b)
    assert relation.rule_verdict == "RECONCILED_BY_CONTEXT"
    assert relation.axis == "scope"
    assert relation.rule_fired == "envelope_difference_explains"


def test_gate3_subject_granularity_axis():
    a = _claim(subject_key="delhivery_limited", value_num=100.0)
    b = _claim(subject_key="delhivery_limited_group", value_num=150.0)
    relation = reconcile(a, b)
    assert relation.rule_verdict == "RECONCILED_BY_CONTEXT"
    assert relation.axis == "subject_granularity"
