from datetime import date

import pymupdf

from ingest.parse import _blocks_excluding_tables, _parse_pdf_date, _table_rows_to_texts, parse_pdf


def test_parse_pdf_date_standard_format():
    assert _parse_pdf_date("D:20220516175837+05'30'") == date(2022, 5, 16)


def test_parse_pdf_date_no_timezone():
    assert _parse_pdf_date("D:20251121151037") == date(2025, 11, 21)


def test_parse_pdf_date_none_or_malformed():
    assert _parse_pdf_date(None) is None
    assert _parse_pdf_date("") is None
    assert _parse_pdf_date("not a date") is None


def test_table_rows_to_texts_pairs_header_with_each_row():
    header = ["Item", "2023-24", "2024-25"]
    rows = [["Trade Balance", "-6.7", "-7.9"], ["Current Account Balance", "-0.7", "-1.3"]]
    texts = _table_rows_to_texts(header, rows)
    assert texts == [
        "Item: Trade Balance | 2023-24: -6.7 | 2024-25: -7.9",
        "Item: Current Account Balance | 2023-24: -0.7 | 2024-25: -1.3",
    ]


def test_table_rows_to_texts_skips_empty_cells():
    header = ["Item", "2023-24", "2024-25"]
    rows = [["Trade Balance", "", "-7.9"]]
    texts = _table_rows_to_texts(header, rows)
    assert texts == ["Item: Trade Balance | 2024-25: -7.9"]


def test_parse_pdf_extracts_prose_and_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Revenue from operations stood at Rs 100 crore in FY2024.")
    doc.set_metadata({"creationDate": "D:20240517174601+05'30'"})
    doc.save(str(pdf_path))
    doc.close()

    parsed = parse_pdf(str(pdf_path))
    assert parsed.page_count == 1
    assert parsed.doc_date == date(2024, 5, 17)
    assert any("Revenue from operations" in b.text for b in parsed.blocks)
    assert all(b.kind in ("prose", "table") for b in parsed.blocks)


def test_parse_pdf_prose_excludes_table_region_text(tmp_path):
    # Regression test for the duplicate table+prose bug: parse_pdf() used to emit
    # page.get_text() (the full linearized page, including every table flattened
    # to raw text) as a single prose block IN ADDITION TO the structurally
    # extracted table rows. That meant every table cell was seen twice by the
    # LLM -- once correctly structured, once flattened exactly the way the
    # project is designed to avoid -- causing duplicate facts and wasted budget.
    #
    # This test builds a real PDF with a paragraph of prose PLUS a real 2x2
    # grid (drawn lines + insert_textbox) that pymupdf's find_tables() actually
    # detects as a table, and asserts the resulting prose text contains the
    # prose sentence but NOT the table's cell text.
    pdf_path = tmp_path / "with_table.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Revenue from operations stood at Rs 100 crore in FY2024.")

    x0, y0 = 72, 150
    col_w, row_h = 100, 20
    for r in range(3):
        page.draw_line((x0, y0 + r * row_h), (x0 + 2 * col_w, y0 + r * row_h))
    for c in range(3):
        page.draw_line((x0 + c * col_w, y0), (x0 + c * col_w, y0 + 2 * row_h))
    page.insert_textbox(pymupdf.Rect(x0, y0, x0 + col_w, y0 + row_h), "Item")
    page.insert_textbox(pymupdf.Rect(x0 + col_w, y0, x0 + 2 * col_w, y0 + row_h), "Value")
    page.insert_textbox(pymupdf.Rect(x0, y0 + row_h, x0 + col_w, y0 + 2 * row_h), "TotalIncomeMarker")
    page.insert_textbox(pymupdf.Rect(x0 + col_w, y0 + row_h, x0 + 2 * col_w, y0 + 2 * row_h), "99999")

    doc.save(str(pdf_path))
    doc.close()

    # Sanity check: find_tables() must actually detect this synthetic table in
    # this pymupdf install, otherwise the test below would pass vacuously.
    check_doc = pymupdf.open(str(pdf_path))
    detected_tables = check_doc[0].find_tables().tables
    assert len(detected_tables) == 1, "synthetic table not detected by find_tables() -- test setup invalid"
    check_doc.close()

    parsed = parse_pdf(str(pdf_path))
    table_blocks = [b for b in parsed.blocks if b.kind == "table"]
    prose_blocks = [b for b in parsed.blocks if b.kind == "prose"]

    assert any("TotalIncomeMarker" in b.text or "99999" in b.text for b in table_blocks)
    assert any("Revenue from operations" in b.text for b in prose_blocks)
    # The table's cell text must NOT also appear in the prose text -- that
    # would mean the same fact was linearized into prose in addition to being
    # structurally extracted as a table row.
    assert not any("TotalIncomeMarker" in b.text for b in prose_blocks)
    assert not any("99999" in b.text for b in prose_blocks)


def test_blocks_excluding_tables_skips_overlapping_and_keeps_others():
    # Focused unit test for the factored-out helper, using plain tuples/Rects
    # so it doesn't depend on find_tables() ever detecting anything -- a
    # belt-and-suspenders check independent of the real-PDF test above.
    table_rect = pymupdf.Rect(72, 150, 272, 190)
    blocks = [
        (72.0, 60.0, 360.0, 75.0, "Revenue from operations stood at Rs 100 crore.\n", 0, 0),
        (72.0, 150.0, 200.0, 165.0, "Item\nValue\n", 1, 0),
        (72.0, 170.0, 190.0, 185.0, "Revenue\n100\n", 2, 0),
        (72.0, 400.0, 300.0, 420.0, "A second unrelated paragraph.\n", 3, 0),
    ]
    kept = _blocks_excluding_tables(blocks, [table_rect])
    assert kept == [
        "Revenue from operations stood at Rs 100 crore.",
        "A second unrelated paragraph.",
    ]


def test_blocks_excluding_tables_with_no_tables_keeps_everything():
    blocks = [
        (0.0, 0.0, 100.0, 10.0, "First block.\n", 0, 0),
        (0.0, 20.0, 100.0, 30.0, "Second block.\n", 1, 0),
    ]
    kept = _blocks_excluding_tables(blocks, [])
    assert kept == ["First block.", "Second block."]
