from datetime import date

from reason.periods import PeriodInfo, parse_period_label, compare_periods


def test_parse_fiscal_year_four_digit():
    info = parse_period_label("FY2024")
    assert info.start == date(2023, 4, 1)
    assert info.end == date(2024, 3, 31)


def test_parse_fiscal_year_two_digit():
    info = parse_period_label("FY24")
    assert info.start == date(2023, 4, 1)
    assert info.end == date(2024, 3, 31)


def test_parse_fiscal_year_range_notation():
    info = parse_period_label("2024-25")
    assert info.start == date(2024, 4, 1)
    assert info.end == date(2025, 3, 31)


def test_parse_fiscal_year_range_with_prefix():
    info = parse_period_label("FY2024-25")
    assert info.start == date(2024, 4, 1)
    assert info.end == date(2025, 3, 31)


def test_parse_quarter():
    q4_fy24 = parse_period_label("Q4 FY24")
    assert q4_fy24.start == date(2024, 1, 1)
    assert q4_fy24.end == date(2024, 3, 31)

    q1_fy25 = parse_period_label("Q1 FY25")
    assert q1_fy25.start == date(2024, 4, 1)
    assert q1_fy25.end == date(2024, 6, 30)


def test_parse_half_year():
    h1 = parse_period_label("H1 FY25")
    assert h1.start == date(2024, 4, 1)
    assert h1.end == date(2024, 9, 30)


def test_parse_nine_months_ended():
    info = parse_period_label("nine months period ended December 31, 2021")
    assert info.start == date(2021, 4, 1)
    assert info.end == date(2021, 12, 31)


def test_parse_nine_months_ended_short_form():
    info = parse_period_label("9M ended 2021-12-31")
    assert info.start == date(2021, 4, 1)
    assert info.end == date(2021, 12, 31)


def test_parse_calendar_year():
    info = parse_period_label("CY2023")
    assert info.start == date(2023, 1, 1)
    assert info.end == date(2023, 12, 31)


def test_parse_bare_four_digit_year_as_calendar_year():
    info = parse_period_label("2023")
    assert info.start == date(2023, 1, 1)
    assert info.end == date(2023, 12, 31)


def test_parse_unknown_returns_none_none():
    info = parse_period_label("as of the reporting date")
    assert info.start is None
    assert info.end is None


def test_parse_none_label():
    info = parse_period_label(None)
    assert info.start is None
    assert info.end is None


def test_compare_periods_same():
    a = PeriodInfo(date(2024, 4, 1), date(2025, 3, 31))
    b = PeriodInfo(date(2024, 4, 1), date(2025, 3, 31))
    assert compare_periods(a, b) == "same"


def test_compare_periods_nested():
    fy = PeriodInfo(date(2021, 4, 1), date(2022, 3, 31))
    nine_months = PeriodInfo(date(2021, 4, 1), date(2021, 12, 31))
    assert compare_periods(fy, nine_months) == "nested"
    assert compare_periods(nine_months, fy) == "nested"


def test_compare_periods_overlapping():
    a = PeriodInfo(date(2024, 1, 1), date(2024, 6, 30))
    b = PeriodInfo(date(2024, 4, 1), date(2024, 9, 30))
    assert compare_periods(a, b) == "overlapping"


def test_compare_periods_disjoint():
    a = PeriodInfo(date(2023, 4, 1), date(2024, 3, 31))
    b = PeriodInfo(date(2024, 4, 1), date(2025, 3, 31))
    assert compare_periods(a, b) == "disjoint"


def test_compare_periods_unknown_when_either_side_incomplete():
    a = PeriodInfo(None, date(2024, 3, 31))
    b = PeriodInfo(date(2023, 4, 1), date(2024, 3, 31))
    assert compare_periods(a, b) == "unknown"
