# reason/explain.py
import hashlib

from app.models import Claim, Relation
from extract.llm import LLMClient, cache_key_for
from extract.prompts import EXPLANATION_PROMPT_TEMPLATE, EXPLANATION_PROMPT_VERSION

INTERESTING_VERDICTS = ("CONTRADICTS", "RECONCILED_BY_CONTEXT")


def relation_content_hash(claim_a: Claim, claim_b: Claim, rule_fired: str) -> str:
    parts = sorted([
        f"{claim_a.subject_key}|{claim_a.measure_key}|{claim_a.evidence_quote}",
        f"{claim_b.subject_key}|{claim_b.measure_key}|{claim_b.evidence_quote}",
    ])
    raw = "||".join(parts) + f"||{rule_fired}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def explain_relation(llm: LLMClient, relation: Relation, claim_a: Claim, claim_b: Claim) -> Relation:
    if relation.rule_verdict not in INTERESTING_VERDICTS:
        relation.final_verdict = relation.rule_verdict
        return relation

    prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        rule_verdict=relation.rule_verdict, rule_fired=relation.rule_fired, axis=relation.axis,
        claim_a_measure=claim_a.measure,
        claim_a_value=claim_a.value_num if claim_a.value_num is not None else claim_a.value_text,
        claim_a_unit=claim_a.unit_raw or "", claim_a_period=claim_a.period_label or "unknown",
        claim_a_quote=claim_a.evidence_quote,
        claim_b_measure=claim_b.measure,
        claim_b_value=claim_b.value_num if claim_b.value_num is not None else claim_b.value_text,
        claim_b_unit=claim_b.unit_raw or "", claim_b_period=claim_b.period_label or "unknown",
        claim_b_quote=claim_b.evidence_quote,
    )
    content_hash = relation_content_hash(claim_a, claim_b, relation.rule_fired)
    cache_key = cache_key_for(content_hash, EXPLANATION_PROMPT_VERSION)

    try:
        result = llm.complete_json(prompt, cache_key)
    except Exception:
        relation.explanation = None
        relation.final_verdict = relation.rule_verdict
        relation.llm_overrode = False
        return relation

    relation.explanation = result.get("explanation")
    override = result.get("override")
    if override in ("RECONCILED_BY_CONTEXT", "CONTRADICTS") and override != relation.rule_verdict:
        relation.final_verdict = override
        relation.llm_overrode = True
        relation.override_reason = result.get("override_reason")
    else:
        relation.final_verdict = relation.rule_verdict
        relation.llm_overrode = False
    return relation
