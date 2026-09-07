# tests/test_prompts.py
from extract.prompts import (
    EXTRACTION_PROMPT_TEMPLATE, EXPLANATION_PROMPT_TEMPLATE,
    EXTRACTION_PROMPT_VERSION, EXPLANATION_PROMPT_VERSION, build_registry_block,
)


def test_build_registry_block_empty():
    assert "none yet" in build_registry_block([]).lower()


def test_build_registry_block_non_empty():
    block = build_registry_block([
        {"measure_key": "revenue_from_operations", "label": "Revenue from Operations"},
        {"measure_key": "real_gdp_growth", "label": "Real GDP Growth"},
    ])
    assert "revenue_from_operations" in block
    assert "real_gdp_growth" in block


def test_extraction_prompt_renders_without_keyerror():
    rendered = EXTRACTION_PROMPT_TEMPLATE.format(
        registry_block=build_registry_block([]), chunk_text="Revenue was Rs 100 crore in FY24.",
    )
    assert "Revenue was Rs 100 crore" in rendered
    assert "evidence_quote" in rendered


def test_explanation_prompt_renders_without_keyerror():
    rendered = EXPLANATION_PROMPT_TEMPLATE.format(
        rule_verdict="CONTRADICTS", rule_fired="large_gap_despite_vintage_gap", axis="vintage",
        claim_a_measure="Current Account Deficit", claim_a_value=-1.3, claim_a_unit="per cent",
        claim_a_period="2024-25", claim_a_quote="quote a",
        claim_b_measure="Current Account Deficit", claim_b_value=-0.6, claim_b_unit="per cent",
        claim_b_period="FY2024/25", claim_b_quote="quote b",
    )
    assert "CONTRADICTS" in rendered
    assert "quote a" in rendered and "quote b" in rendered


def test_prompt_versions_are_short_strings():
    assert isinstance(EXTRACTION_PROMPT_VERSION, str) and len(EXTRACTION_PROMPT_VERSION) < 10
    assert isinstance(EXPLANATION_PROMPT_VERSION, str) and len(EXPLANATION_PROMPT_VERSION) < 10
