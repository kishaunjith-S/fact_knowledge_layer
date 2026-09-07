from ingest.score import score_chunk


def test_dense_financial_text_scores_higher_than_boilerplate():
    dense = ("Revenue from operations on consolidated basis for FY24 stood at "
              "Rs 81,415.38 million, a growth of 12.68 per cent over Rs 72,253.01 million.")
    boilerplate = ("This page is intentionally left blank. Please refer to the "
                   "table of contents for further navigation instructions.")
    assert score_chunk(dense) > score_chunk(boilerplate)


def test_table_row_text_scores_higher_than_prose_with_no_numbers():
    table_row = "Item: Current Account Balance | 2023-24: -0.7 | 2024-25: -1.3"
    prose = "The board of directors met to discuss the overall strategic direction of the company."
    assert score_chunk(table_row) > score_chunk(prose)


def test_score_is_non_negative():
    assert score_chunk("") >= 0.0
    assert score_chunk("no numbers or keywords here at all") >= 0.0


def test_keywords_increase_score():
    with_keyword = "The director resigned from the board effective immediately."
    without_keyword = "The weather today is quite pleasant and sunny."
    assert score_chunk(with_keyword) > score_chunk(without_keyword)


def test_information_rich_long_table_outranks_tiny_dense_fragment():
    # A long multi-year table is the highest-value content in a filing, but
    # linear length normalization used to bury it beneath a tiny fragment that
    # happens to pack two figures into 20 characters -- the exact reason the
    # real RBI Balance-of-Payments table ranked ~#168 of 628 chunks.
    long_table = "External sector - Balance of Payments\n" + "\n".join(
        f"{name}: 2021-22 {v}.2 | 2022-23 {v}.5 | 2023-24 {v}.9 | 2024-25 {v + 1}.1 per cent of GDP"
        for name, v in [
            ("Current account balance", 1), ("Merchandise trade balance", 6),
            ("Net services", 4), ("Net income", 3), ("Capital account", 8),
        ]
    )
    tiny_dense = "loss Rs 17,833.04 million"
    assert score_chunk(long_table) > score_chunk(tiny_dense)


def test_fiscal_year_headers_increase_score():
    with_fy = "Revenue was FY2024: 100"
    without_fy = "Revenue was 2024: 100"
    assert score_chunk(with_fy) > score_chunk(without_fy)
