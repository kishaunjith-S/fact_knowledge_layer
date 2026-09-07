import re
from dataclasses import dataclass

_SCALE_WORDS = (
    ("crore", 1e7),
    ("billion", 1e9),
    ("million", 1e6),
    ("lakh", 1e5),
    ("lac", 1e5),
    ("thousand", 1e3),
)
# Abbreviations that appear in real filings and earnings decks ("₹ Cr",
# "Rs mn", "USD bn"). Matched as whole tokens only, so "cr" never fires
# from inside a word like "accrued".
_SCALE_ABBREV = {
    "cr": 1e7, "crs": 1e7,
    "mn": 1e6, "mln": 1e6,
    "bn": 1e9, "bln": 1e9,
}
_CURRENCY_MARKERS = ("₹", "rs.", "rs ", "inr", "rupee")
_PERCENT_TOKENS = ("%", "percent", "per cent", "pct")


@dataclass(frozen=True)
class UnitInfo:
    unit_dim: str | None
    scale: float


def parse_unit(unit_raw: str | None) -> UnitInfo:
    if not unit_raw or not unit_raw.strip():
        return UnitInfo(unit_dim=None, scale=1.0)
    text = unit_raw.strip().lower()

    if text in _PERCENT_TOKENS:
        return UnitInfo(unit_dim="percent", scale=1.0)
    if "bps" in text or "basis point" in text:
        return UnitInfo(unit_dim="percent", scale=0.01)
    if text == "rs" or "rs." in text or "rs " in text or "₹" in text or "inr" in text or "rupee" in text:
        for word, mult in _SCALE_WORDS:
            if word in text:
                return UnitInfo(unit_dim="currency_inr", scale=mult)
        for token in re.split(r"[^a-z]+", text):
            if token in _SCALE_ABBREV:
                return UnitInfo(unit_dim="currency_inr", scale=_SCALE_ABBREV[token])
        return UnitInfo(unit_dim="currency_inr", scale=1.0)
    if text in ("ratio", "x", "times"):
        return UnitInfo(unit_dim="ratio", scale=1.0)
    if text in ("count", "number", "units", "nos", "no."):
        return UnitInfo(unit_dim="count", scale=1.0)
    return UnitInfo(unit_dim=None, scale=1.0)


def normalize_value(value: float, unit_info: UnitInfo) -> float:
    return value * unit_info.scale


def _looks_rounded(value: float) -> bool:
    """Heuristic: treats a value written with <=3 significant figures as
    'rounded', widening the agreement tolerance per spec Section 5."""
    if value == 0:
        return True
    text = f"{abs(value):.10g}"
    digits = text.replace(".", "").lstrip("0")
    return len(digits) <= 3


def values_agree(
    a_value: float, a_unit: UnitInfo, b_value: float, b_unit: UnitInfo,
    tolerance: float = 0.01, rounded_tolerance: float = 0.05,
) -> tuple[bool, float]:
    """Compare two values for agreement within tolerance, accounting for unit scaling.

    When unit dimensions match, scales both values before comparison. When dimensions
    differ, compares raw unscaled values—this comparison is NOT dimensionally meaningful
    (e.g., 5.0 percent vs 5.0 crore will spuriously agree if magnitudes happen to be close).

    This function performs NO dimension-compatibility check. Callers must gate on
    unit_dim equality before calling if dimensional correctness is required.

    Returns: tuple of (agree: bool, delta_pct: float) where delta_pct is the percent
    difference between scaled (or raw) values.
    """
    # Only scale if dimensions match; otherwise compare raw values
    if a_unit.unit_dim == b_unit.unit_dim:
        va = normalize_value(a_value, a_unit)
        vb = normalize_value(b_value, b_unit)
    else:
        va = a_value
        vb = b_value
    denom = max(abs(va), abs(vb), 1e-9)
    delta_pct = abs(va - vb) / denom
    tol = rounded_tolerance if (_looks_rounded(a_value) or _looks_rounded(b_value)) else tolerance
    return delta_pct <= tol, delta_pct
