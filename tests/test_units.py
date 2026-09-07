from reason.units import UnitInfo, parse_unit, normalize_value, values_agree


def test_parse_percent():
    assert parse_unit("per cent") == UnitInfo(unit_dim="percent", scale=1.0)
    assert parse_unit("%") == UnitInfo(unit_dim="percent", scale=1.0)
    assert parse_unit("percent") == UnitInfo(unit_dim="percent", scale=1.0)


def test_parse_bps():
    assert parse_unit("bps") == UnitInfo(unit_dim="percent", scale=0.01)


def test_parse_currency_scales():
    assert parse_unit("₹ million") == UnitInfo(unit_dim="currency_inr", scale=1e6)
    assert parse_unit("Rs. crore") == UnitInfo(unit_dim="currency_inr", scale=1e7)
    assert parse_unit("INR lakh") == UnitInfo(unit_dim="currency_inr", scale=1e5)
    assert parse_unit("₹ billion") == UnitInfo(unit_dim="currency_inr", scale=1e9)
    assert parse_unit("Rs.") == UnitInfo(unit_dim="currency_inr", scale=1.0)


def test_parse_currency_abbreviations():
    # earnings decks write "Cr" / "mn" / "bn", not the full scale word
    assert parse_unit("₹ Cr") == UnitInfo(unit_dim="currency_inr", scale=1e7)
    assert parse_unit("Rs Cr") == UnitInfo(unit_dim="currency_inr", scale=1e7)
    assert parse_unit("INR mn") == UnitInfo(unit_dim="currency_inr", scale=1e6)
    assert parse_unit("₹ bn") == UnitInfo(unit_dim="currency_inr", scale=1e9)


def test_currency_abbreviation_not_matched_inside_a_word():
    # "cr" must not fire from a substring of a longer token
    assert parse_unit("₹ accrued balance").scale == 1.0


def test_parse_count_and_ratio():
    assert parse_unit("count") == UnitInfo(unit_dim="count", scale=1.0)
    assert parse_unit("ratio") == UnitInfo(unit_dim="ratio", scale=1.0)


def test_parse_none_and_unknown():
    assert parse_unit(None) == UnitInfo(unit_dim=None, scale=1.0)
    assert parse_unit("gigawatts") == UnitInfo(unit_dim=None, scale=1.0)


def test_normalize_value():
    unit = UnitInfo(unit_dim="currency_inr", scale=1e6)
    assert normalize_value(81415.38, unit) == 81415.38e6


def test_values_agree_delhivery_revenue_case():
    # Annual Report: Rs 81,415.38 million. Earnings deck: Rs 8,142 crore.
    a_unit = parse_unit("₹ million")
    b_unit = parse_unit("₹ crore")
    agree, delta_pct = values_agree(81415.38, a_unit, 8142, b_unit)
    assert agree is True
    assert delta_pct < 0.001


def test_values_agree_exact_gdp_growth_case():
    unit = parse_unit("per cent")
    agree, delta_pct = values_agree(6.5, unit, 6.5, unit)
    assert agree is True
    assert delta_pct == 0.0


def test_values_agree_cad_magnitude_cap_case():
    # RBI -1.3% vs IMF -0.6% for FY2024-25 -- confirmed genuine contradiction (spec Section 11 case 2)
    unit = parse_unit("per cent")
    agree, delta_pct = values_agree(-1.3, unit, -0.6, unit)
    assert agree is False
    assert delta_pct > 0.5


def test_values_agree_widened_tolerance_for_rounded_numbers():
    unit = parse_unit("per cent")
    agree_tight, _ = values_agree(6.50, unit, 6.60, unit, tolerance=0.01, rounded_tolerance=0.05)
    assert agree_tight is True  # 6.5 and 6.6 both look rounded (<=3 sig figs) -> widened tolerance applies


def test_values_disagree_incommensurable_dims_not_compared_here():
    # values_agree does raw math regardless of dim; the dimension GATE lives in reconcile.py
    percent = parse_unit("per cent")
    currency = parse_unit("₹ crore")
    agree, delta_pct = values_agree(5.0, percent, 5.0, currency)
    assert agree is True  # values_agree only compares magnitudes; callers must gate on unit_dim first
