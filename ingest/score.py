import re

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")
_CURRENCY_RE = re.compile(r"₹|Rs\.?|INR|crore|lakh|million|billion", re.IGNORECASE)
_PERCENT_RE = re.compile(r"%|per\s*cent|percent", re.IGNORECASE)
_KEYWORD_RE = re.compile(
    r"revenue|profit|loss|deficit|surplus|growth|inflation|director|resigned|"
    r"appointed|address|registered office|rating|debt|reserve|deficit|balance",
    re.IGNORECASE,
)
# A fiscal-year token (FY24, FY2024, 2024-25) is a strong marker of a
# period-anchored data row -- the rows the extractor most needs to see.
_FY_HEADER_RE = re.compile(r"\bFY\s?\d{2,4}\b|\b20\d\d[-/]\d{2}\b", re.IGNORECASE)


def score_chunk(text: str) -> float:
    if not text.strip():
        return 0.0
    length = max(len(text), 1)
    numbers = len(_NUMBER_RE.findall(text))
    currency = len(_CURRENCY_RE.findall(text))
    percent = len(_PERCENT_RE.findall(text))
    keywords = len(_KEYWORD_RE.findall(text))
    fy_headers = len(_FY_HEADER_RE.findall(text))
    weighted_hits = (
        numbers * 1.0 + currency * 2.0 + percent * 2.0
        + keywords * 1.5 + fy_headers * 3.0
    )
    # Sub-linear length normalization. Pure per-character density lets a 20-char
    # fragment with two figures outrank a long, genuinely fact-rich multi-year
    # table; sqrt keeps density meaningful while not punishing structured tables
    # -- the highest-value content -- for being long.
    density = weighted_hits / ((length / 200.0) ** 0.5)
    return round(density, 4)
