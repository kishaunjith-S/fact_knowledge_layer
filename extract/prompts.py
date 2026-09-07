# extract/prompts.py
EXTRACTION_PROMPT_VERSION = "v1"
EXPLANATION_PROMPT_VERSION = "v1"

EXTRACTION_PROMPT_TEMPLATE = """You are extracting verifiable facts from a document excerpt for a fact knowledge layer.

Known measure keys already in use (prefer reusing one of these over minting a new one):
{registry_block}

For each fact-bearing statement in the TEXT below, emit one JSON object with these fields:
- subject: the entity the fact is about, as named in the text
- subject_key: a snake_case slug for the subject (e.g. "delhivery_limited")
- measure: the quantity or attribute being stated, as named in the text
- measure_key: a snake_case slug for the measure; reuse one from the list above whenever it fits, mint a new one only when nothing fits
- value_type: one of "numeric", "text", "date", "boolean"
- value_num: the numeric value if value_type is "numeric", else null
- value_text: the text/date/boolean value as a string if value_type is not "numeric", else null
- unit_raw: the unit exactly as written (e.g. "Rs. crore", "per cent"), or null if none
- period_label: the time period the value covers, exactly as written (e.g. "FY2024", "nine months ended December 31, 2021"), or null if none stated
- basis: an object of scope qualifiers explicitly stated in the text (e.g. {{"consolidated": true}}, {{"restated": true}}, {{"projected": true}}); omit keys not stated, use {{}} if none
- evidence_quote: a short verbatim quote (under 40 words) copied EXACTLY from the TEXT below that supports this fact
- confidence: your honest confidence in this extraction, 0.0 to 1.0

Rules:
- evidence_quote MUST be an exact substring of the TEXT below. Do not paraphrase it.
- Omit a field (use null) rather than guessing its value.
- Extract only facts that are actually stated in the TEXT; do not infer facts from outside knowledge.
- Return a JSON array of these objects. Return [] if the TEXT has no extractable facts.

TEXT:
{chunk_text}

Return only the JSON array, no other text."""

EXPLANATION_PROMPT_TEMPLATE = """You are reviewing an automated fact-comparison system's output for a knowledge layer.

For the relation below, a deterministic rule produced a verdict. Write a short (1-3 sentence) explanation of the relationship for a human reader, grounded only in the two evidence quotes given. You may override the verdict to "RECONCILED_BY_CONTEXT" or "CONTRADICTS" if the evidence clearly supports a different call than the rule made, but only do so with a specific, stated reason citing the evidence.

Rule verdict: {rule_verdict}
Rule fired: {rule_fired}
Axis: {axis}

Claim A: {claim_a_measure} = {claim_a_value} {claim_a_unit} ({claim_a_period}), source: "{claim_a_quote}"
Claim B: {claim_b_measure} = {claim_b_value} {claim_b_unit} ({claim_b_period}), source: "{claim_b_quote}"

Return a single JSON object: {{"explanation": "...", "override": null or "RECONCILED_BY_CONTEXT" or "CONTRADICTS", "override_reason": null or "..."}}
Return only the JSON object, no other text."""


def build_registry_block(top_measures: list[dict]) -> str:
    if not top_measures:
        return "(none yet -- mint keys as needed)"
    return "\n".join(f"- {m['measure_key']}: {m['label']}" for m in top_measures)
