import re

from app.models import Claim

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = _NON_ALNUM_RE.sub("_", text)
    text = text.strip("_")
    return text or "unknown"


def group_claims(claims: list[Claim]) -> dict[tuple[str, str], list[Claim]]:
    groups: dict[tuple[str, str], list[Claim]] = {}
    for claim in claims:
        key = (claim.subject_key, claim.measure_key)
        groups.setdefault(key, []).append(claim)
    return groups
