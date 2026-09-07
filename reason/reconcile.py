# reason/reconcile.py
from datetime import date

from app.models import Claim, Relation
from reason.periods import PeriodInfo, compare_periods
from reason.units import UnitInfo, values_agree

MAGNITUDE_CAP = 0.5
VINTAGE_GAP_DAYS = 90


def basis_differs(a_basis: dict, b_basis: dict) -> tuple[bool, str | None]:
    a_basis = a_basis or {}
    b_basis = b_basis or {}
    for key in sorted(set(a_basis) & set(b_basis)):
        if a_basis[key] != b_basis[key]:
            return True, key
    for flag in ("restated", "projected"):
        a_has, b_has = flag in a_basis, flag in b_basis
        if a_has != b_has:
            asserting_side = a_basis if a_has else b_basis
            if asserting_side.get(flag) is True:
                return True, flag
    return False, None


def subject_granularity_differs(a_key: str, b_key: str) -> bool:
    if a_key == b_key:
        return False
    a_tokens, b_tokens = a_key.split("_"), b_key.split("_")
    shorter, longer = (a_tokens, b_tokens) if len(a_tokens) <= len(b_tokens) else (b_tokens, a_tokens)
    return longer[: len(shorter)] == shorter


def vintage_gap_days(a_vintage: date | None, b_vintage: date | None) -> int | None:
    if a_vintage is None or b_vintage is None:
        return None
    return abs((a_vintage - b_vintage).days)


def _relation(a: Claim, b: Claim, verdict: str, axis: str | None, rule: str,
              delta_pct: float | None = None) -> Relation:
    return Relation(
        id="", claim_a_id=a.id, claim_b_id=b.id,
        rule_verdict=verdict, final_verdict=verdict,
        axis=axis, rule_fired=rule, delta_pct=delta_pct,
    )


def _values_agree(a: Claim, b: Claim) -> tuple[bool, float | None]:
    if a.value_type == "numeric":
        return values_agree(a.value_num, UnitInfo(a.unit_dim, a.unit_scale),
                             b.value_num, UnitInfo(b.unit_dim, b.unit_scale))
    a_text = (a.value_text or "").strip().lower()
    b_text = (b.value_text or "").strip().lower()
    return a_text == b_text, None


def _primary_axis(period_rel: str, basis_axis_differs: bool, a: Claim, b: Claim) -> str:
    if period_rel != "same":
        return "period"
    if basis_axis_differs:
        return "scope"
    gap = vintage_gap_days(a.doc_vintage, b.doc_vintage)
    if gap is not None and gap >= VINTAGE_GAP_DAYS:
        return "vintage"
    if subject_granularity_differs(a.subject_key, b.subject_key):
        return "subject_granularity"
    return "vintage"


def reconcile(a: Claim, b: Claim) -> Relation:
    # GATE 0: incommensurable -- different value types, or (for numeric
    # claims) different unit dimensions. This MUST run before any call to
    # values_agree(): that function has no dimension-compatibility check of
    # its own (see its docstring in reason/units.py) and will silently
    # compare raw unscaled numbers across dimensions if not gated first.
    if a.value_type != b.value_type:
        return _relation(a, b, "INSUFFICIENT_CONTEXT", axis=None, rule="incommensurable_value_types")

    if a.value_type == "numeric":
        if a.unit_dim != b.unit_dim:
            return _relation(a, b, "INSUFFICIENT_CONTEXT", axis="unit", rule="incommensurable_units")
        if a.value_num is None or b.value_num is None:
            return _relation(a, b, "INSUFFICIENT_CONTEXT", axis=None, rule="incomplete_envelope")

    # GATE 1: incomplete envelope -- can't reason about period at all.
    if a.period_end is None or b.period_end is None:
        return _relation(a, b, "INSUFFICIENT_CONTEXT", axis=None, rule="incomplete_envelope")

    period_rel = compare_periods(PeriodInfo(a.period_start, a.period_end), PeriodInfo(b.period_start, b.period_end))
    if period_rel == "unknown":
        return _relation(a, b, "INSUFFICIENT_CONTEXT", axis="period", rule="unclear_period_relationship")

    agree, delta_pct = _values_agree(a, b)
    differs, _basis_axis_key = basis_differs(a.basis, b.basis)
    same_subject_granularity = not subject_granularity_differs(a.subject_key, b.subject_key)

    # GATE 2: identical envelopes -- period, scope (basis) and subject
    # granularity all line up, so any value difference must be judged as
    # agreement/disagreement/vintage-explained rather than attributed to a
    # differing envelope.
    if period_rel == "same" and not differs and same_subject_granularity:
        if agree:
            same_unit = a.value_type != "numeric" or (a.unit_dim == b.unit_dim and a.unit_scale == b.unit_scale)
            rule = "exact_envelope_match" if same_unit else "unit_normalized_match"
            return _relation(a, b, "CORROBORATES", axis=None, rule=rule, delta_pct=delta_pct)

        gap = vintage_gap_days(a.doc_vintage, b.doc_vintage)
        if gap is not None and gap >= VINTAGE_GAP_DAYS:
            # Non-numeric mismatches (delta_pct is None) have no magnitude to cap against --
            # a status change over time (e.g. director active -> resigned) is inherently
            # plausible across a vintage gap, so it defaults to reconciled rather than
            # contradicted. Numeric mismatches are capped per spec Section 5.
            if delta_pct is None or delta_pct <= MAGNITUDE_CAP:
                return _relation(a, b, "RECONCILED_BY_CONTEXT", axis="vintage",
                                  rule="later_vintage_revision", delta_pct=delta_pct)
            return _relation(a, b, "CONTRADICTS", axis="vintage",
                              rule="large_gap_despite_vintage_gap", delta_pct=delta_pct)
        return _relation(a, b, "CONTRADICTS", axis=None, rule="same_envelope_value_mismatch", delta_pct=delta_pct)

    # GATE 3: envelopes differ -- attribute the difference to the
    # highest-priority axis that actually differs (period > scope > vintage >
    # subject_granularity).
    primary_axis = _primary_axis(period_rel, differs, a, b)
    if agree:
        return _relation(a, b, "RECONCILED_BY_CONTEXT", axis=primary_axis,
                          rule="coincident_values_differing_envelope", delta_pct=delta_pct)
    return _relation(a, b, "RECONCILED_BY_CONTEXT", axis=primary_axis,
                      rule="envelope_difference_explains", delta_pct=delta_pct)
