import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class PeriodInfo:
    start: date | None
    end: date | None


def _fiscal_year_bounds(fy_end_year: int) -> tuple[date, date]:
    """Indian fiscal year FY<end_year> runs April 1 of (end_year-1) to March 31 of end_year."""
    return date(fy_end_year - 1, 4, 1), date(fy_end_year, 3, 31)


def _to_full_year(two_or_four_digit: str) -> int:
    year = int(two_or_four_digit)
    return year if year > 100 else 2000 + year


_FY_RANGE_RE = re.compile(r"^(?:FY)?\s*(\d{4})\s*[-/]\s*(\d{2,4})$", re.IGNORECASE)
_FY_SINGLE_RE = re.compile(r"^FY\s*(\d{2,4})$", re.IGNORECASE)
_QUARTER_RE = re.compile(r"^Q([1-4])\s*FY\s*(\d{2,4})$", re.IGNORECASE)
_HALF_RE = re.compile(r"^H([12])\s*FY\s*(\d{2,4})$", re.IGNORECASE)
_CY_RE = re.compile(r"^CY\s*(\d{4})$", re.IGNORECASE)
_BARE_YEAR_RE = re.compile(r"^(\d{4})$")
_MONTHS_ENDED_RE = re.compile(
    r"(nine|six|three|twelve|9|6|3|12)\s*(?:months?|M)\s*(?:period\s*)?ended\s+(.+)$",
    re.IGNORECASE,
)
_MONTH_WORDS = {
    "nine": 9, "six": 6, "three": 3, "twelve": 12,
    "9": 9, "6": 6, "3": 3, "12": 12,
}
_DATE_FORMATS = ("%B %d, %Y", "%d %B %Y", "%Y-%m-%d", "%d-%m-%Y")
_QUARTER_STARTS = {1: 4, 2: 7, 3: 10, 4: 1}  # calendar month each Indian-FY quarter starts in


def _parse_date_flex(text: str) -> date | None:
    text = text.strip().rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _fiscal_year_start_for_date(d: date) -> date:
    return date(d.year, 4, 1) if d.month >= 4 else date(d.year - 1, 4, 1)


def parse_period_label(label: str | None) -> PeriodInfo:
    if not label or not label.strip():
        return PeriodInfo(None, None)
    text = label.strip()

    months_ended = _MONTHS_ENDED_RE.search(text)
    if months_ended:
        count = _MONTH_WORDS.get(months_ended.group(1).lower())
        end_date = _parse_date_flex(months_ended.group(2))
        if count is not None and end_date is not None:
            fy_start = _fiscal_year_start_for_date(end_date)
            month_index = (end_date.year - fy_start.year) * 12 + (end_date.month - fy_start.month) + 1
            if month_index == count:
                return PeriodInfo(fy_start, end_date)
            start_month = end_date.month - count + 1
            start_year = end_date.year
            while start_month <= 0:
                start_month += 12
                start_year -= 1
            return PeriodInfo(date(start_year, start_month, 1), end_date)

    match = _QUARTER_RE.match(text)
    if match:
        quarter, fy = int(match.group(1)), _to_full_year(match.group(2))
        fy_start_year = fy - 1
        start_month = _QUARTER_STARTS[quarter]
        start_year = fy_start_year if quarter != 4 else fy
        start = date(start_year, start_month, 1)
        end_month = start_month + 2
        end_year = start_year
        if end_month > 12:
            end_month -= 12
            end_year += 1
        last_day = calendar.monthrange(end_year, end_month)[1]
        return PeriodInfo(start, date(end_year, end_month, last_day))

    match = _HALF_RE.match(text)
    if match:
        half, fy = int(match.group(1)), _to_full_year(match.group(2))
        fy_start, fy_end = _fiscal_year_bounds(fy)
        if half == 1:
            return PeriodInfo(fy_start, date(fy_start.year, 9, 30))
        return PeriodInfo(date(fy_start.year, 10, 1), fy_end)

    match = _FY_RANGE_RE.match(text)
    if match:
        start_year = _to_full_year(match.group(1))
        return PeriodInfo(date(start_year, 4, 1), date(start_year + 1, 3, 31))

    match = _FY_SINGLE_RE.match(text)
    if match:
        return PeriodInfo(*_fiscal_year_bounds(_to_full_year(match.group(1))))

    match = _CY_RE.match(text)
    if match:
        year = int(match.group(1))
        return PeriodInfo(date(year, 1, 1), date(year, 12, 31))

    match = _BARE_YEAR_RE.match(text)
    if match:
        year = int(match.group(1))
        return PeriodInfo(date(year, 1, 1), date(year, 12, 31))

    return PeriodInfo(None, None)


def compare_periods(a: PeriodInfo, b: PeriodInfo) -> str:
    if a.start is None or a.end is None or b.start is None or b.end is None:
        return "unknown"
    if a.start == b.start and a.end == b.end:
        return "same"
    a_contains_b = a.start <= b.start and a.end >= b.end
    b_contains_a = b.start <= a.start and b.end >= a.end
    if a_contains_b or b_contains_a:
        return "nested"
    if a.start <= b.end and b.start <= a.end:
        return "overlapping"
    return "disjoint"
