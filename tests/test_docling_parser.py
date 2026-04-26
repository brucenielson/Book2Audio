"""Tests for the DoclingParser class."""

import pytest
from unittest.mock import MagicMock
from docling_core.types.doc.document import SectionHeaderItem, TextItem, DocItemLabel
from docling_core.types import DoclingDocument
from parsers.docling_parser import DoclingParser, _FootnoteContext
from text_cleaner import TextCleaner


# --- Fixtures ---

def make_doc_item(spec, label: str, text: str, page_no: int = 1) -> MagicMock:
    """Create a mock DocItem with the given label, text, and page number."""
    item = MagicMock(spec=spec)
    item.label = label
    item.text = text
    prov = MagicMock()
    prov.page_no = page_no
    prov.bbox = MagicMock()
    prov.bbox.height = 10.0
    prov.charspan = (0, 10)
    item.prov = [prov]
    return item


def make_text_item(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock regular text item."""
    return make_doc_item(TextItem, DocItemLabel.TEXT.value, text, page_no)


def make_section_header(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock section header item."""
    return make_doc_item(SectionHeaderItem, DocItemLabel.SECTION_HEADER.value, text, page_no)


def make_footnote(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock footnote item."""
    return make_doc_item(TextItem, DocItemLabel.FOOTNOTE.value, text, page_no)


def make_page_header(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock page header item."""
    return make_doc_item(TextItem, DocItemLabel.PAGE_HEADER.value, text, page_no)


def make_page_footer(text: str, page_no: int = 1) -> MagicMock:
    """Create a mock page footer item."""
    return make_doc_item(TextItem, DocItemLabel.PAGE_FOOTER.value, text, page_no)


def make_parser(texts: list,
                meta_data: dict | None = None,
                min_paragraph_size: int = 0,
                start_page: int | None = None,
                end_page: int | None = None,
                include_notes: bool = True,
                cleaner: TextCleaner | None = None,
                min_footnote_chars: int = 100) -> DoclingParser:
    """Create a DoclingParser with a mocked DoclingDocument."""
    doc = MagicMock(spec=DoclingDocument)
    doc.name = "test_doc"
    doc.texts = texts
    return DoclingParser(source=doc, meta_data=meta_data or {}, min_paragraph_size=min_paragraph_size,
                         start_page=start_page, end_page=end_page, include_footnotes=include_notes,
                         llm_cleaner=cleaner, min_footnote_chars=min_footnote_chars)


def make_ctx(
    prev_text_candidate: bool = False,
    text_seen_this_page: bool = False,
    found_note_this_page: bool = False,
    single_line_height: float = 10.0,
    median_chars_per_line: float = 50.0,
) -> _FootnoteContext:
    """Create a _FootnoteContext with sensible defaults for unit testing."""
    return _FootnoteContext(
        prev_text_candidate=prev_text_candidate,
        text_seen_this_page=text_seen_this_page,
        found_note_this_page=found_note_this_page,
        single_line_height=single_line_height,
        median_chars_per_line=median_chars_per_line,
    )


def make_sized_text_item(text: str, page_no: int = 1,
                         charspan_length: int = 10,
                         bbox_height: float = 10.0) -> MagicMock:
    """Create a TEXT item with configurable charspan and bbox height for H2 testing."""
    item = make_doc_item(TextItem, DocItemLabel.TEXT.value, text, page_no)
    item.prov[0].charspan = (0, charspan_length)
    item.prov[0].bbox.height = bbox_height
    return item


# --- TestIsFootnote ---

class TestIsFootnote:

    # --- Already-labeled FOOTNOTE ---

    def test_labeled_footnote_returns_true(self) -> None:
        """Items Docling already labeled as FOOTNOTE must always return True."""
        parser = make_parser([])
        assert parser._is_footnote(make_footnote("1 Already labeled."), make_ctx()) is True

    def test_labeled_footnote_ignores_ctx(self) -> None:
        """FOOTNOTE label is sufficient on its own — context state is irrelevant."""
        parser = make_parser([])
        ctx = make_ctx(prev_text_candidate=False, text_seen_this_page=False,
                       found_note_this_page=False)
        assert parser._is_footnote(make_footnote("1 Already labeled."), ctx) is True

    # --- Guard: label is not FOOTNOTE and item is not digit-start TEXT ---

    def test_text_starting_with_letter_returns_false(self) -> None:
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("Regular body text."), make_ctx()) is False

    def test_section_header_with_digit_start_returns_false(self) -> None:
        """SECTION_HEADER label must fail the guard even when text starts with a digit."""
        parser = make_parser([])
        ctx = make_ctx(prev_text_candidate=True, found_note_this_page=True)
        assert parser._is_footnote(make_section_header("1. Introduction"), ctx) is False

    def test_page_header_returns_false(self) -> None:
        parser = make_parser([])
        assert parser._is_footnote(make_page_header("1 Page Header"), make_ctx()) is False

    def test_page_footer_returns_false(self) -> None:
        parser = make_parser([])
        assert parser._is_footnote(make_page_footer("1 Page Footer"), make_ctx()) is False

    def test_list_item_returns_false(self) -> None:
        """LIST_ITEM label must fail the guard even with digit-start text."""
        item = make_doc_item(TextItem, DocItemLabel.LIST_ITEM.value, "1 list entry")
        parser = make_parser([])
        assert parser._is_footnote(item, make_ctx(prev_text_candidate=True)) is False

    def test_empty_text_returns_false(self) -> None:
        """Empty string is falsy — guard bails before any heuristic is checked."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item(""), make_ctx()) is False

    def test_text_starting_with_space_returns_false(self) -> None:
        """A leading space before a digit is not a digit-start — guard must not pass."""
        parser = make_parser([])
        ctx = make_ctx(prev_text_candidate=True)
        assert parser._is_footnote(make_text_item(" 1 Leading space."), ctx) is False

    # --- H1: digit-start TEXT following mid-sentence body text ---

    def test_h1_digit_alpha_after_mid_sentence_returns_true(self) -> None:
        parser = make_parser([])
        item = make_text_item("1 This is an unlabelled footnote.")
        assert parser._is_footnote(item, make_ctx(prev_text_candidate=True)) is True

    def test_h1_pure_number_with_prev_candidate_returns_false(self) -> None:
        """Index entries like '183-84' contain no alpha — H1 must not fire."""
        parser = make_parser([])
        item = make_text_item("183-84")
        assert parser._is_footnote(item, make_ctx(prev_text_candidate=True)) is False

    def test_h1_alpha_without_prev_candidate_returns_false(self) -> None:
        """Alpha alone is not enough — H1 also requires prev_text_candidate."""
        parser = make_parser([])
        item = make_text_item("1 Some text.")
        assert parser._is_footnote(item, make_ctx(prev_text_candidate=False)) is False

    # --- H2: small font, preceded by body text on this page ---
    # is_small_text: chars_per_line = charspan / (height / single_line_height)
    # Fires when chars_per_line > median * 1.25
    # Setup: charspan=200, height=10, single_line_height=5 → chars_per_line=100
    #   small:     median=50  → 100 > 62.5  → True
    #   not small: median=200 → 100 > 250   → False

    def test_h2_small_text_with_body_seen_returns_true(self) -> None:
        """Long digit-start item in small font, preceded by body text → H2 fires."""
        text = "1" + "a" * 99   # len=100, digit-start, has alpha
        item = make_sized_text_item(text, charspan_length=200, bbox_height=10.0)
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=50.0)
        assert parser._is_footnote(item, ctx) is True

    def test_h2_fires_without_alpha(self) -> None:
        """H2 has no alpha requirement — a long digit-only small-font item qualifies."""
        text = "1" + "0" * 99   # len=100, digit-start, no alpha
        item = make_sized_text_item(text, charspan_length=200, bbox_height=10.0)
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=50.0)
        assert parser._is_footnote(item, ctx) is True

    def test_h2_text_below_threshold_returns_false(self) -> None:
        """Text shorter than min_footnote_chars must not trigger H2."""
        text = "1 short"   # len < 100
        item = make_sized_text_item(text, charspan_length=200, bbox_height=10.0)
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=50.0)
        assert parser._is_footnote(item, ctx) is False

    def test_h2_no_body_text_seen_returns_false(self) -> None:
        """H2 must not fire if no body text has appeared yet on this page."""
        text = "1" + "a" * 99
        item = make_sized_text_item(text, charspan_length=200, bbox_height=10.0)
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(text_seen_this_page=False, single_line_height=5.0,
                       median_chars_per_line=50.0)
        assert parser._is_footnote(item, ctx) is False

    def test_h2_normal_font_size_returns_false(self) -> None:
        """H2 must not fire when the item's font size matches the document norm."""
        text = "1" + "a" * 99
        item = make_sized_text_item(text, charspan_length=200, bbox_height=10.0)
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=200.0)  # high median → not small
        assert parser._is_footnote(item, ctx) is False

    # --- H3: propagation after a footnote has been seen on this page ---

    def test_h3_digit_alpha_after_note_on_page_returns_true(self) -> None:
        parser = make_parser([])
        item = make_text_item("2 Continuation of a footnote.")
        assert parser._is_footnote(item, make_ctx(found_note_this_page=True)) is True

    def test_h3_pure_number_after_note_returns_false(self) -> None:
        """No alpha — H3 must not fire even when a note has been seen on the page."""
        parser = make_parser([])
        item = make_text_item("2")
        assert parser._is_footnote(item, make_ctx(found_note_this_page=True)) is False

    def test_h3_alpha_no_prior_note_returns_false(self) -> None:
        """Alpha alone is not enough — H3 also requires found_note_this_page."""
        parser = make_parser([])
        item = make_text_item("2 Some text.")
        assert parser._is_footnote(item, make_ctx(found_note_this_page=False)) is False

    # --- No heuristic fires ---

    def test_digit_only_text_all_ctx_false_returns_false(self) -> None:
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("42"), make_ctx()) is False

    def test_digit_alpha_text_all_ctx_false_returns_false(self) -> None:
        """Has alpha and digit-start but no context conditions met — must return False."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("1 Some text."), make_ctx()) is False


# --- TestGetProcessedTexts ---

class TestGetProcessedTexts:
    def test_separates_regular_and_footnotes(self) -> None:
        texts = [
            make_text_item("Regular text."),
            make_footnote("Footnote text."),
        ]
        parser = make_parser(texts)
        regular, notes = parser._get_processed_texts()
        assert len(regular) == 1
        assert len(notes) == 1

    def test_skips_too_short_items(self) -> None:
        texts = [
            make_text_item("Hi"),
            make_text_item("This is a longer sentence."),
        ]
        parser = make_parser(texts)
        regular, notes = parser._get_processed_texts()
        assert len(regular) == 1
        assert regular[0].text == "This is a longer sentence."

    def test_regular_texts_before_notes(self) -> None:
        texts = [
            make_footnote("Footnote."),
            make_text_item("Regular text."),
        ]
        parser = make_parser(texts)
        regular, notes = parser._get_processed_texts()
        assert regular[0].text == "Regular text."
        assert notes[0].text == "Footnote."

    def test_empty_document(self) -> None:
        parser = make_parser([])
        regular, notes = parser._get_processed_texts()
        assert regular == []
        assert notes == []


# --- TestRun ---

class TestRun:
    def test_basic_paragraph(self) -> None:
        texts = [make_text_item("This is a complete sentence.")]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert len(docs) == 1
        assert "This is a complete sentence." in docs[0]

    def test_meta_contains_expected_keys(self) -> None:
        texts = [make_text_item("This is a complete sentence.")]
        parser = make_parser(texts, meta_data={"source": "test"})
        docs, meta = parser.run()
        assert meta[0]["source"] == "test"
        assert "section_name" in meta[0]
        assert "page_#" in meta[0]
        assert "paragraph_#" in meta[0]

    def test_section_header_becomes_paragraph(self) -> None:
        texts = [make_section_header("Chapter One")]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert any("Chapter One" in d for d in docs)

    def test_section_header_flushes_accumulated_paragraph(self) -> None:
        texts = [
            make_text_item("First sentence without end"),
            make_section_header("Chapter Two"),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert any("First sentence without end" in d for d in docs)
        assert any("Chapter Two" in d for d in docs)

    def test_skips_page_header(self) -> None:
        texts = [
            make_page_header("Page Header"),
            make_text_item("Real content."),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert all("Page Header" not in d for d in docs)

    def test_skips_page_footer(self) -> None:
        texts = [
            make_text_item("Real content."),
            make_page_footer("Page Footer"),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert all("Page Footer" not in d for d in docs)

    def test_include_notes_true(self) -> None:
        texts = [
            make_text_item("Main text."),
            make_footnote("Footnote content."),
        ]
        parser = make_parser(texts, include_notes=True)
        docs, meta = parser.run()
        assert any("Footnote content." in d for d in docs)

    def test_include_notes_false(self) -> None:
        texts = [
            make_text_item("Main text."),
            make_footnote("Footnote content."),
        ]
        parser = make_parser(texts, include_notes=False)
        docs, meta = parser.run()
        assert all("Footnote content." not in d for d in docs)

    def test_digit_start_after_incomplete_sentence_classified_as_note(self) -> None:
        # Preceding text is long and doesn't end with punctuation → footnote heuristic fires
        preceding = "A" * 100
        texts = [
            make_text_item(preceding),
            make_text_item("1 This is an unlabelled footnote reference."),
        ]
        parser = make_parser(texts, include_notes=False, min_footnote_chars=100)
        docs, meta = parser.run()
        assert all("unlabelled footnote" not in d for d in docs)

    def test_digit_start_after_short_text_not_classified_as_note(self) -> None:
        # Preceding text is too short (below min_footnote_chars) → heuristic must not fire
        short_preceding = "Button Gwinnett Lyman Hall"  # name list, no punctuation, but short
        texts = [
            make_text_item(short_preceding),
            make_text_item("6 The Declaration of Independence of The United States of America"),
        ]
        parser = make_parser(texts, include_notes=False, min_footnote_chars=100)
        docs, meta = parser.run()
        assert any("Declaration of Independence" in d for d in docs)

    def test_digit_start_with_no_alpha_not_classified_as_note(self) -> None:
        # Pure number/punctuation continuation (e.g. "183-84" from an index entry)
        # has no alphabetic content → must never be classified as a footnote
        long_no_punct = "Jehovah's Witnesses, 1, 48, 50, 160-61, 221 justificationism, xvi, 60, 124-28, 130,"
        texts = [
            make_text_item(long_no_punct),
            make_text_item("183-84"),
        ]
        parser = make_parser(texts, include_notes=False, min_footnote_chars=100)
        docs, meta = parser.run()
        assert any("183-84" in d for d in docs)

    def test_start_page_filters_early_pages(self) -> None:
        texts = [
            make_text_item("Page one content.", page_no=1),
            make_text_item("Page two content.", page_no=2),
        ]
        parser = make_parser(texts, start_page=2)
        docs, meta = parser.run()
        assert all("Page one content." not in d for d in docs)
        assert any("Page two content." in d for d in docs)

    def test_end_page_filters_later_pages(self) -> None:
        texts = [
            make_text_item("Page one content.", page_no=1),
            make_text_item("Page two content.", page_no=2),
        ]
        parser = make_parser(texts, end_page=1)
        docs, meta = parser.run()
        assert any("Page one content." in d for d in docs)
        assert all("Page two content." not in d for d in docs)

    def test_page_range_inclusive(self) -> None:
        texts = [
            make_text_item("Page one.", page_no=1),
            make_text_item("Page two.", page_no=2),
            make_text_item("Page three.", page_no=3),
        ]
        parser = make_parser(texts, start_page=1, end_page=2)
        docs, meta = parser.run()
        assert any("Page one." in d for d in docs)
        assert any("Page two." in d for d in docs)
        assert all("Page three." not in d for d in docs)

    def test_accumulates_short_paragraphs(self) -> None:
        texts = [
            make_text_item("First sentence."),
            make_text_item("Second sentence."),
        ]
        parser = make_parser(texts, min_paragraph_size=100)
        docs, meta = parser.run()
        assert len(docs) == 1
        assert "First sentence." in docs[0]
        assert "Second sentence." in docs[0]

    def test_paragraph_number_in_meta(self) -> None:
        texts = [
            make_text_item("First paragraph."),
            make_text_item("Second paragraph."),
        ]
        parser = make_parser(texts, min_paragraph_size=0)
        docs, meta = parser.run()
        assert meta[0]["paragraph_#"] == "1"
        assert meta[1]["paragraph_#"] == "2"

    def test_section_name_in_meta(self) -> None:
        texts = [
            make_section_header("Chapter One"),
            make_text_item("Content here."),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        # The paragraph after the section header should have the section name
        content_meta = next(m for m in meta if m["section_name"] == "Chapter One")
        assert content_meta is not None

    def test_empty_document_returns_empty(self) -> None:
        parser = make_parser([])
        docs, meta = parser.run()
        assert docs == []
        assert meta == []


# --- TestIntegration ---

class TestIntegration:
    @pytest.mark.integration
    def test_mislabelled_footnote_dropped_by_cleaner(self) -> None:
        """A footnote mislabeled as body text should be identified and dropped by the LLM cleaner."""
        texts = [
            make_text_item(
                "Others have found very similar defection rates in various minor religious sects.1",
                page_no=1
            ),
            make_text_item(
                "1 This ignores the interesting question of whether the defectors have given up "
                "all the beliefs in the doctrines of the movement they have quit.",
                page_no=1
            ),
        ]
        parser = make_parser(texts, cleaner=TextCleaner(temperature=0), include_notes=False)
        docs, meta = parser.run()
        assert any("religious sects" in d for d in docs)
        assert all("This ignores the interesting question" not in d for d in docs)
