from ingest.parse import PageBlock
from ingest.chunk import content_hash, blocks_to_chunks


def test_content_hash_is_stable_and_whitespace_insensitive():
    h1 = content_hash("Revenue   grew  by 6.5 per cent")
    h2 = content_hash("Revenue grew by 6.5 per cent")
    assert h1 == h2
    assert h1 != content_hash("something else entirely")


def test_content_hash_is_case_insensitive():
    assert content_hash("Revenue") == content_hash("revenue")


def test_blocks_to_chunks_table_block_stays_whole():
    block = PageBlock(page_no=1, kind="table", text="Item: Revenue | 2024-25: 100", char_start=0)
    chunks = blocks_to_chunks("doc1", [block])
    assert len(chunks) == 1
    assert chunks[0].kind == "table"
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].page_no == 1
    assert chunks[0].extracted is False


def test_blocks_to_chunks_splits_long_prose_on_paragraph_boundaries():
    paragraph = "Sentence. " * 50  # ~500 chars
    text = "\n\n".join([paragraph] * 5)  # ~2500 chars, well over max_prose_chars
    block = PageBlock(page_no=2, kind="prose", text=text, char_start=0)
    chunks = blocks_to_chunks("doc1", [block], max_prose_chars=1000)
    assert len(chunks) > 1
    assert all(c.kind == "prose" for c in chunks)
    assert "".join(c.text for c in chunks).replace("\n\n", "") != ""


def test_blocks_to_chunks_short_prose_stays_one_chunk():
    block = PageBlock(page_no=1, kind="prose", text="A short paragraph of text.", char_start=0)
    chunks = blocks_to_chunks("doc1", [block], max_prose_chars=1500)
    assert len(chunks) == 1


def test_blocks_to_chunks_skips_blank_blocks():
    block = PageBlock(page_no=1, kind="prose", text="   \n\n  ", char_start=0)
    chunks = blocks_to_chunks("doc1", [block])
    assert chunks == []


def test_single_long_paragraph_with_no_double_newline_is_still_split():
    # Regression test: max_prose_chars was inert when a single "paragraph" (no
    # "\n\n" inside it at all -- e.g. one dense uninterrupted block of text)
    # exceeded max_chars on its own. _split_prose split only on "\n\n", so one
    # long paragraph came back as a single chunk far bigger than max_chars.
    # After the Fix 1 change to parse.py, "\n\n"-joined blocks are real
    # paragraph boundaries in practice, but _split_prose must still defend
    # against a single overlong block/paragraph on its own.
    word = "lorem "  # 6 chars
    single_paragraph = word * 500  # 3000 chars, no "\n\n" anywhere inside it
    block = PageBlock(page_no=1, kind="prose", text=single_paragraph, char_start=0)
    max_chars = 1000

    chunks = blocks_to_chunks("doc1", [block], max_prose_chars=max_chars)

    assert len(chunks) > 1
    # Every chunk should be close to (not wildly exceeding) max_chars. Allow
    # a little slack for splitting on whitespace boundaries rather than a
    # hard character cut.
    for c in chunks:
        assert len(c.text) <= max_chars * 1.1, f"chunk of {len(c.text)} chars exceeds cap"
    # No content should be lost.
    rejoined = "".join(c.text for c in chunks)
    assert rejoined.replace(" ", "") == single_paragraph.replace(" ", "")
