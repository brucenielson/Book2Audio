"""Tests for the DoclingParser class."""

import pytest
from unittest.mock import MagicMock
from docling_core.types.doc.document import SectionHeaderItem, TextItem, DocItemLabel
from docling_core.types import DoclingDocument
from parsers.docling_parser import DoclingParser, _FootnoteContext
from text_cleaner import TextCleaner

from conftest import TEST_LLM_MODEL


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
    prov.bbox.t = 0.0
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


def make_section_header_at(text: str, page_no: int = 1,
                            bbox_t: float = 0.0, bbox_height: float = 10.0) -> MagicMock:
    """Create a mock section header with explicit bbox.t and height."""
    item = make_section_header(text, page_no)
    item.prov[0].bbox.t = bbox_t
    item.prov[0].bbox.height = bbox_height
    return item


def make_page_header_at(text: str, page_no: int = 1, bbox_t: float = 0.0) -> MagicMock:
    """Create a mock page header with explicit bbox.t."""
    item = make_page_header(text, page_no)
    item.prov[0].bbox.t = bbox_t
    return item


def make_parser(texts: list,
                meta_data: dict | None = None,
                min_paragraph_size: int = 0,
                start_page: int | None = None,
                end_page: int | None = None,
                include_notes: bool = True,
                cleaner: TextCleaner | None = None,
                min_footnote_chars: int = 100,
                page_labels: dict[int, str] | None = None,
                skip_front_matter: bool = False) -> DoclingParser:
    """Create a DoclingParser with a mocked DoclingDocument."""
    doc = MagicMock(spec=DoclingDocument)
    doc.name = "test_doc"
    doc.texts = texts
    return DoclingParser(source=doc, meta_data=meta_data or {}, min_paragraph_size=min_paragraph_size,
                         start_page=start_page, end_page=end_page, include_footnotes=include_notes,
                         llm_cleaner=cleaner, min_footnote_chars=min_footnote_chars,
                         page_labels=page_labels, skip_front_matter=skip_front_matter)


def make_ctx(
    prev_text_candidate: bool = False,
    text_seen_this_page: bool = False,
    found_note_this_page: bool = False,
    single_line_height: float = 10.0,
    median_chars_per_line: float = 50.0,
    header_top_y: float | None = None,
    median_page_height: float = 0.0,
) -> _FootnoteContext:
    """Create a _FootnoteContext with sensible defaults for unit testing."""
    return _FootnoteContext(
        prev_text_candidate=prev_text_candidate,
        text_seen_this_page=text_seen_this_page,
        found_note_this_page=found_note_this_page,
        single_line_height=single_line_height,
        median_chars_per_line=median_chars_per_line,
        header_top_y=header_top_y,
        median_page_height=median_page_height,
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

    # --- H4: digit(s) immediately followed by uppercase letter ---

    def test_h4_single_digit_uppercase_returns_true(self) -> None:
        """'3See' pattern: single digit immediately followed by uppercase → footnote unconditionally."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("3See my Poverty of Historicism."), make_ctx()) is True

    def test_h4_two_digits_uppercase_returns_true(self) -> None:
        """Two digits immediately followed by uppercase → footnote unconditionally."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("14Cf. the earlier discussion."), make_ctx()) is True

    def test_h4_three_digits_not_caught(self) -> None:
        """Three or more digits before letter should not trigger H4."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("183See something."), make_ctx()) is False

    def test_h4_digit_lowercase_not_caught(self) -> None:
        """Lowercase alpha after digit does not trigger H4 — avoids ordinals like '1st'."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("1st place goes to"), make_ctx()) is False

    def test_h4_digit_space_uppercase_not_caught(self) -> None:
        """Space between digit and letter means H4 does not fire — uses normal H1/H2/H3 path."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("3 See my text."), make_ctx()) is False


# --- TestIsInPageRange ---

class TestIsInPageRange:

    def test_no_range_always_returns_true(self) -> None:
        parser = make_parser([])
        assert parser._is_in_page_range(1) is True
        assert parser._is_in_page_range(999) is True

    def test_none_page_no_always_returns_true(self) -> None:
        parser = make_parser([], start_page=5, end_page=10)
        assert parser._is_in_page_range(None) is True

    def test_start_page_filters_earlier_pages(self) -> None:
        parser = make_parser([], start_page=5)
        assert parser._is_in_page_range(4) is False
        assert parser._is_in_page_range(5) is True
        assert parser._is_in_page_range(6) is True

    def test_end_page_filters_later_pages(self) -> None:
        parser = make_parser([], end_page=10)
        assert parser._is_in_page_range(9) is True
        assert parser._is_in_page_range(10) is True
        assert parser._is_in_page_range(11) is False

    def test_both_bounds_inclusive(self) -> None:
        parser = make_parser([], start_page=3, end_page=7)
        assert parser._is_in_page_range(2) is False
        assert parser._is_in_page_range(3) is True
        assert parser._is_in_page_range(5) is True
        assert parser._is_in_page_range(7) is True
        assert parser._is_in_page_range(8) is False


# --- TestExtractChunks ---

class TestExtractChunks:

    def test_regular_texts_become_chunks(self) -> None:
        texts = [make_text_item("Body text.")]
        parser = make_parser([])
        chunks = parser._extract_chunks(texts, [])
        assert len(chunks) == 1
        assert chunks[0].text == "Body text."

    def test_notes_excluded_when_include_notes_false(self) -> None:
        notes = [make_text_item("Footnote text.")]
        parser = make_parser([], include_notes=False)
        chunks = parser._extract_chunks([], notes)
        assert chunks == []

    def test_notes_included_when_include_notes_true(self) -> None:
        notes = [make_text_item("Footnote text.")]
        parser = make_parser([], include_notes=True)
        chunks = parser._extract_chunks([], notes)
        assert len(chunks) == 1
        assert chunks[0].text == "Footnote text."

    def test_page_range_filters_out_of_range_items(self) -> None:
        texts = [
            make_text_item("Page 2 text.", page_no=2),
            make_text_item("Page 5 text.", page_no=5),
        ]
        parser = make_parser([], start_page=5, end_page=10)
        chunks = parser._extract_chunks(texts, [])
        assert len(chunks) == 1
        assert chunks[0].text == "Page 5 text."

    def test_chunk_meta_contains_page_number(self) -> None:
        texts = [make_text_item("Text.", page_no=42)]
        parser = make_parser([])
        chunks = parser._extract_chunks(texts, [])
        assert chunks[0].meta['page_#'] == '42'

    def test_chunk_meta_contains_physical_page_number(self) -> None:
        texts = [make_text_item("Text.", page_no=42)]
        parser = make_parser([])
        chunks = parser._extract_chunks(texts, [])
        assert chunks[0].meta['physical_page_#'] == '42'

    def test_chunk_label_matches_item_label(self) -> None:
        texts = [make_text_item("Text.")]
        parser = make_parser([])
        chunks = parser._extract_chunks(texts, [])
        assert chunks[0].label == DocItemLabel.TEXT




# --- TestCalibrateHeaderTopY ---

class TestCalibrateHeaderTopY:
    def test_no_page_headers_returns_none(self) -> None:
        """No PAGE_HEADER items in doc → returns None."""
        texts = [make_text_item("Body text.")]
        parser = make_parser(texts)
        assert parser._calibrate_header_top_y() is None

    def test_single_page_header_returns_its_t(self) -> None:
        """A single PAGE_HEADER → its bbox.t is returned."""
        header = make_page_header_at("Running Head", page_no=1, bbox_t=20.0)
        texts = [header]
        parser = make_parser(texts)
        assert parser._calibrate_header_top_y() == pytest.approx(20.0)

    def test_page_header_after_body_text_still_counted(self) -> None:
        """PAGE_HEADER that appears after body text in doc.texts is still used.
        Docling does not guarantee page headers are listed first — only the label matters."""
        body = make_text_item("Some body text.", page_no=1)
        header = make_page_header_at("Running Head", page_no=1, bbox_t=20.0)
        texts = [body, header]  # header comes after body in doc.texts
        parser = make_parser(texts)
        assert parser._calibrate_header_top_y() == pytest.approx(20.0)

    def test_multiple_page_headers_returns_median(self) -> None:
        """Multiple PAGE_HEADERs → returns median bbox.t."""
        h1 = make_page_header_at("Head 1", page_no=1, bbox_t=20.0)
        h2 = make_page_header_at("Head 2", page_no=2, bbox_t=22.0)
        h3 = make_page_header_at("Head 3", page_no=3, bbox_t=18.0)
        texts = [h1, h2, h3]
        parser = make_parser(texts)
        result = parser._calibrate_header_top_y()
        assert result == pytest.approx(20.0)  # median of [18, 20, 22]


# --- TestIsPageHeader ---

class TestIsPageHeader:

    # --- Path A: validated PAGE_HEADER reference available ---

    def test_path_a_section_header_at_reference_y_returns_true(self) -> None:
        """SECTION_HEADER at reference y, single-line, no prior body text → True."""
        header = make_section_header_at("Running Head", bbox_t=20.0, bbox_height=10.0)
        ctx = make_ctx(header_top_y=20.0, single_line_height=10.0, text_seen_this_page=False)
        assert DoclingParser._is_page_header(header, ctx) is True

    def test_path_a_y_within_tolerance_returns_true(self) -> None:
        """bbox.t within ± single_line_height of reference → True."""
        header = make_section_header_at("Running Head", bbox_t=26.0, bbox_height=10.0)
        ctx = make_ctx(header_top_y=20.0, single_line_height=10.0, text_seen_this_page=False)
        assert DoclingParser._is_page_header(header, ctx) is True

    def test_path_a_y_outside_tolerance_returns_false(self) -> None:
        """bbox.t more than single_line_height from reference → False."""
        header = make_section_header_at("Chapter One", bbox_t=200.0, bbox_height=10.0)
        ctx = make_ctx(header_top_y=20.0, single_line_height=10.0, text_seen_this_page=False)
        assert DoclingParser._is_page_header(header, ctx) is False

    def test_path_a_not_section_header_returns_false(self) -> None:
        """Non-SECTION_HEADER items are never running heads."""
        text = make_text_item("Some text")
        text.prov[0].bbox.t = 20.0
        ctx = make_ctx(header_top_y=20.0, single_line_height=10.0, text_seen_this_page=False)
        assert DoclingParser._is_page_header(text, ctx) is False

    def test_path_a_suppressed_even_after_body_text(self) -> None:
        """A section header at the reference y-coordinate is still a running head even when
        body text appeared before it in Docling's text ordering. Docling does not guarantee
        that mislabeled running heads appear before body text in doc.texts — the visual
        position (bbox.t) is the reliable discriminator, not doc.texts order."""
        header = make_section_header_at("REBUTTAL", bbox_t=20.0, bbox_height=10.0)
        ctx = make_ctx(header_top_y=20.0, single_line_height=10.0, text_seen_this_page=True)
        assert DoclingParser._is_page_header(header, ctx) is True

    def test_path_a_multi_line_returns_false(self) -> None:
        """Multi-line item at reference y is a real header, not a running head."""
        header = make_section_header_at("Running Head", bbox_t=20.0, bbox_height=30.0)
        ctx = make_ctx(header_top_y=20.0, single_line_height=10.0, text_seen_this_page=False)
        assert DoclingParser._is_page_header(header, ctx) is False

    # --- Path B: no validated PAGE_HEADER reference ---

    def test_path_b_at_top_of_page_returns_true(self) -> None:
        """Section header in top 15% of page → True when no reference available.

        Docling PDFs use BOTTOMLEFT coordinates: bbox.t increases going up, so
        a header near the top of the page has a large bbox.t (close to page height)."""
        header = make_section_header_at("Running Head", bbox_t=900.0, bbox_height=10.0)
        ctx = make_ctx(header_top_y=None, median_page_height=1000.0,
                       single_line_height=10.0, text_seen_this_page=False)
        assert DoclingParser._is_page_header(header, ctx) is True

    def test_path_b_not_at_top_returns_false(self) -> None:
        """Section header in middle of page → False in Path B."""
        header = make_section_header_at("Chapter One", bbox_t=500.0, bbox_height=10.0)
        ctx = make_ctx(header_top_y=None, median_page_height=1000.0,
                       single_line_height=10.0, text_seen_this_page=False)
        assert DoclingParser._is_page_header(header, ctx) is False

    def test_path_b_suppressed_even_after_body_text(self) -> None:
        """Path B also uses position alone — doc.texts ordering is not reliable."""
        header = make_section_header_at("Running Head", bbox_t=900.0, bbox_height=10.0)
        ctx = make_ctx(header_top_y=None, median_page_height=1000.0,
                       single_line_height=10.0, text_seen_this_page=True)
        assert DoclingParser._is_page_header(header, ctx) is True

    def test_path_b_no_page_height_returns_false(self) -> None:
        """With no page height info, Path B must not fire (division by zero guard)."""
        header = make_section_header_at("Running Head", bbox_t=50.0, bbox_height=10.0)
        ctx = make_ctx(header_top_y=None, median_page_height=0.0,
                       single_line_height=10.0, text_seen_this_page=False)
        assert DoclingParser._is_page_header(header, ctx) is False

    def test_path_b_multi_line_returns_false(self) -> None:
        """Multi-line item near top of page is a real chapter header, not a running head."""
        header = make_section_header_at("Running Head", bbox_t=50.0, bbox_height=30.0)
        ctx = make_ctx(header_top_y=None, median_page_height=1000.0,
                       single_line_height=10.0, text_seen_this_page=False)
        assert DoclingParser._is_page_header(header, ctx) is False


# --- TestUpdateTextState ---

class TestUpdateTextState:
    def test_long_mid_sentence_text_sets_prev_text_candidate(self) -> None:
        text = make_text_item("A" * 100)   # long, no sentence-ending punctuation
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx()
        parser._update_text_state(text, ctx)
        assert ctx.prev_text_candidate is True

    def test_short_text_clears_prev_text_candidate(self) -> None:
        text = make_text_item("Short text")
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(prev_text_candidate=True)
        parser._update_text_state(text, ctx)
        assert ctx.prev_text_candidate is False

    def test_sentence_ending_text_clears_prev_text_candidate(self) -> None:
        text = make_text_item("A" * 100 + ".")
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx()
        parser._update_text_state(text, ctx)
        assert ctx.prev_text_candidate is False

    def test_colon_ending_clears_prev_text_candidate(self) -> None:
        """Text ending with ':' clears prev_text_candidate even when long.
        A following digit-start item after a colon is a list continuation, not a footnote."""
        text = make_text_item("A" * 100 + ":")
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx()
        parser._update_text_state(text, ctx)
        assert ctx.prev_text_candidate is False

    def test_non_text_label_does_not_change_prev_text_candidate(self) -> None:
        """Non-TEXT items (section headers etc.) do not affect the H1 footnote gate."""
        header = make_section_header("A Chapter")
        parser = make_parser([])
        ctx = make_ctx(prev_text_candidate=True)
        parser._update_text_state(header, ctx)
        assert ctx.prev_text_candidate is True


# --- TestGetProcessedTexts ---

class TestGetProcessedTexts:
    def test_separates_regular_and_footnotes(self) -> None:
        texts = [
            make_text_item("Regular text."),
            make_footnote("Footnote text."),
        ]
        parser = make_parser(texts)
        classified = parser._get_processed_texts()
        _SKIP = {'footnote', 'page_header', 'too_short'}
        regular = [item for item, label in classified if label not in _SKIP]
        notes = [item for item, label in classified if label == 'footnote']
        assert len(regular) == 1
        assert len(notes) == 1

    def test_skips_too_short_items(self) -> None:
        texts = [
            make_text_item("Hi"),
            make_text_item("This is a longer sentence."),
        ]
        parser = make_parser(texts)
        classified = parser._get_processed_texts()
        assert len(classified) == 2
        assert classified[0][1] == 'too_short'
        assert classified[1][1] != 'too_short'

    def test_document_order_preserved(self) -> None:
        texts = [
            make_footnote("Footnote."),
            make_text_item("Regular text."),
        ]
        parser = make_parser(texts)
        classified = parser._get_processed_texts()
        assert classified[0][0].text == "Footnote."
        assert classified[1][0].text == "Regular text."

    def test_empty_document(self) -> None:
        parser = make_parser([])
        classified = parser._get_processed_texts()
        assert classified == []


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

    def test_section_header_kept_after_sentence_end(self) -> None:
        """A section header preceded by sentence-ending text is retained."""
        texts = [
            make_text_item("First sentence ends here."),
            make_section_header("Chapter Two"),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert any("First sentence ends here." in d for d in docs)
        assert any("Chapter Two" in d for d in docs)

    def test_section_header_skipped_at_top_of_new_page(self) -> None:
        """A single-line section header that is the first item on a new page,
        positioned at the same y-coordinate as a validated running PAGE_HEADER,
        is suppressed as a mislabeled running head."""
        body = "Body text on page one that clearly belongs there."
        texts = [
            make_page_header("Running Head", page_no=1),    # validated: first on page 1, bbox.t=0.0
            make_text_item(body, page_no=1),
            make_section_header_at("Running Head", page_no=2,  # first on page 2, bbox.t=0.0 → matches
                                   bbox_t=0.0, bbox_height=10.0),
        ]
        parser = make_parser(texts)
        docs, meta = parser.run()
        assert any(body in d for d in docs)
        assert all("Running Head" not in d for d in docs)

    def test_section_header_in_content_area_not_suppressed(self) -> None:
        """A section header whose bbox.t places it in the content area (far below the
        header margin) must NOT be suppressed, even when a validated page header exists.
        The calibration derives header_top_y from the page header's bbox.t; the section
        header is far enough below it that the position check leaves it alone."""
        short_colon = "The inference rule has the form:"
        texts = [
            make_page_header("Running Head", page_no=1),                   # calibrated reference at bbox.t=0.0
            make_text_item(short_colon, page_no=1),
            make_section_header_at("1. If P, then Q;", page_no=1,
                                   bbox_t=200.0, bbox_height=10.0),        # content area, far below reference
        ]
        parser = make_parser(texts, min_footnote_chars=100)
        docs, meta = parser.run()
        assert any("1. If P, then Q;" in d for d in docs)

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


# --- TestProcessedTextsFile ---

class TestProcessedTextsFile:
    """_processed_texts.txt must contain every DocItem in document order.
    Reclassified items show 'original_label → new_label:'; unchanged items show just their label."""

    def _make_file_parser(self, texts, tmp_path, **kwargs):
        parser = make_parser(texts, **kwargs)
        parser._file_path = tmp_path / "test.pdf"
        return parser

    def _read_file(self, tmp_path):
        return (tmp_path / "test_doc_processed_texts.txt").read_text(encoding="utf-8")

    def test_suppressed_page_header_appears_in_file(self, tmp_path) -> None:
        """A section header suppressed as a running page header must still be written to the file."""
        texts = [
            make_page_header_at("PH", page_no=1, bbox_t=0.0),   # validates bbox.t=0.0
            make_text_item("Body text.", page_no=1),
            make_section_header_at("Suppressed Head", page_no=2, # first on page 2, same y
                                   bbox_t=0.0, bbox_height=10.0),
        ]
        parser = self._make_file_parser(texts, tmp_path)
        parser.run(generate_text_file=True, annotate_reclassifications=True)
        assert "Suppressed Head" in self._read_file(tmp_path)

    def test_suppressed_page_header_shows_reclassified_label(self, tmp_path) -> None:
        """A suppressed section header must show 'section_header → page_header' on its line."""
        texts = [
            make_page_header_at("PH", page_no=1, bbox_t=0.0),   # validates bbox.t=0.0
            make_text_item("Body text.", page_no=1),
            make_section_header_at("Suppressed Head", page_no=2, # first on page 2, same y
                                   bbox_t=0.0, bbox_height=10.0),
        ]
        parser = self._make_file_parser(texts, tmp_path)
        parser.run(generate_text_file=True, annotate_reclassifications=True)
        content = self._read_file(tmp_path)
        assert any(
            "section_header" in line and "page_header" in line and "Suppressed Head" in line
            for line in content.splitlines()
        )

    def test_reclassified_footnote_shows_arrow_label(self, tmp_path) -> None:
        """A TEXT item reclassified as footnote must show 'text → footnote:' on its line."""
        long_mid = "A" * 100
        texts = [
            make_text_item(long_mid),
            make_text_item("1 This is a citation reference."),
        ]
        parser = self._make_file_parser(texts, tmp_path, min_footnote_chars=100)
        parser.run(generate_text_file=True, annotate_reclassifications=True)
        content = self._read_file(tmp_path)
        assert any(
            "text" in line and "footnote" in line and "citation reference" in line
            for line in content.splitlines()
        )

    def test_unmodified_item_shows_no_arrow(self, tmp_path) -> None:
        """Regular body text that is not reclassified must appear with no → on its line."""
        texts = [make_text_item("Regular body text here.")]
        parser = self._make_file_parser(texts, tmp_path)
        parser.run(generate_text_file=True, annotate_reclassifications=True)
        content = self._read_file(tmp_path)
        body_line = next(l for l in content.splitlines() if "Regular body text here." in l)
        assert "→" not in body_line

    def test_items_appear_in_document_order(self, tmp_path) -> None:
        """All items must appear in document order — footnotes must not be moved to the end."""
        long_mid = "A" * 100
        texts = [
            make_text_item("First body.", page_no=1),
            make_text_item(long_mid, page_no=1),
            make_text_item("1 A citation.", page_no=1),
            make_text_item("Second body.", page_no=2),
        ]
        parser = self._make_file_parser(texts, tmp_path, min_footnote_chars=100)
        parser.run(generate_text_file=True, annotate_reclassifications=False)
        content = self._read_file(tmp_path)
        assert content.index("A citation.") < content.index("Second body.")


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
        parser = make_parser(texts, cleaner=TextCleaner(model=TEST_LLM_MODEL, temperature=0), include_notes=False)
        docs, meta = parser.run()
        assert any("religious sects" in d for d in docs)
        assert all("This ignores the interesting question" not in d for d in docs)


# --- TestFormatPage ---

class TestFormatPage:
    """Tests for DoclingParser._format_page(page_no) -> str.

    When the PDF label matches the physical number, show just one.
    When they differ, show [Page <label> / Page <physical>].
    """

    def test_no_labels_shows_physical_number_only(self) -> None:
        """Without a label table, label falls back to physical — show just the number."""
        parser = make_parser([])
        assert parser._format_page(42) == '42'

    def test_matching_label_shows_single_number(self) -> None:
        """When the PDF label equals the physical number string, show it once."""
        parser = make_parser([], page_labels={4: '5'})   # Docling page 5 → index 4 → '5'
        assert parser._format_page(5) == '5'

    def test_differing_label_shows_both(self) -> None:
        """When label and physical differ, show [Page <label> / Page <physical>]."""
        parser = make_parser([], page_labels={40: '1'})  # Docling page 41 → index 40 → '1'
        assert parser._format_page(41) == '[Page 1 / Page 41]'

    def test_roman_numeral_label_shows_both(self) -> None:
        """Roman numeral front-matter labels always differ from the physical number."""
        parser = make_parser([], page_labels={0: 'i'})
        assert parser._format_page(1) == '[Page i / Page 1]'

    def test_empty_string_label_falls_back_to_physical(self) -> None:
        """pypdfium2 empty-string label is treated as no label — show physical only."""
        parser = make_parser([], page_labels={4: ''})
        assert parser._format_page(5) == '5'


# --- TestPageLabels ---

class TestPageLabels:
    """Tests for PDF page label integration in DoclingParser.

    Docling's page_no is 1-based; pypdfium2 page label indices are 0-based.
    So page_no N maps to label index N-1.
    """

    def test_page_meta_uses_pdf_label_when_available(self) -> None:
        """When labels are provided, chunk metadata uses the label not the physical number."""
        texts = [make_text_item("Body text.", page_no=1)]
        parser = make_parser(texts, page_labels={0: 'i'})
        chunks = parser._extract_chunks(texts, [])
        assert chunks[0].meta['page_#'] == 'i'

    def test_page_meta_falls_back_to_physical_number_without_labels(self) -> None:
        """Without page labels, chunk metadata falls back to the physical page number string."""
        texts = [make_text_item("Body text.", page_no=42)]
        parser = make_parser(texts)
        chunks = parser._extract_chunks(texts, [])
        assert chunks[0].meta['page_#'] == '42'

    def test_empty_string_label_falls_back_to_physical_number(self) -> None:
        """pypdfium2 returns '' for PDFs with no page label table; must still show physical number."""
        texts = [make_text_item("Body text.", page_no=5)]
        parser = make_parser(texts, page_labels={4: ''})   # empty string, not None
        chunks = parser._extract_chunks(texts, [])
        assert chunks[0].meta['page_#'] == '5'

    def test_arabic_label_stored_verbatim_in_meta(self) -> None:
        """Arabic page labels are stored verbatim — the body of a book with 40 front-matter pages."""
        texts = [make_text_item("Body text.", page_no=41)]
        parser = make_parser(texts, page_labels={40: '1'})   # Docling page 41 → index 40 → '1'
        chunks = parser._extract_chunks(texts, [])
        assert chunks[0].meta['page_#'] == '1'

    def test_front_matter_pages_skipped_when_enabled(self) -> None:
        """Pages with Roman numeral labels are excluded when skip_front_matter=True."""
        texts = [
            make_text_item("Front matter text.", page_no=1),
            make_text_item("Body text.", page_no=2),
        ]
        parser = make_parser(texts, page_labels={0: 'i', 1: '1'}, skip_front_matter=True)
        chunks = parser._extract_chunks(texts, [])
        assert len(chunks) == 1
        assert chunks[0].meta['page_#'] == '1'

    def test_front_matter_included_when_skip_front_matter_false(self) -> None:
        """Front matter pages are kept when skip_front_matter=False (the default)."""
        texts = [
            make_text_item("Front matter text.", page_no=1),
            make_text_item("Body text.", page_no=2),
        ]
        parser = make_parser(texts, page_labels={0: 'i', 1: '1'}, skip_front_matter=False)
        chunks = parser._extract_chunks(texts, [])
        assert len(chunks) == 2

    def test_both_page_numbers_present_in_meta(self) -> None:
        """Both the PDF label and the physical page number appear in chunk metadata."""
        texts = [make_text_item("Body text.", page_no=41)]
        parser = make_parser(texts, page_labels={40: '1'})
        chunks = parser._extract_chunks(texts, [])
        assert chunks[0].meta['page_#'] == '1'
        assert chunks[0].meta['physical_page_#'] == '41'

    def test_physical_page_matches_docling_page_no(self) -> None:
        """physical_page_# always reflects Docling's page_no regardless of label."""
        texts = [make_text_item("Front matter.", page_no=5)]
        parser = make_parser(texts, page_labels={4: 'v'})
        chunks = parser._extract_chunks(texts, [])
        assert chunks[0].meta['physical_page_#'] == '5'
        assert chunks[0].meta['page_#'] == 'v'

    def test_multiple_front_matter_pages_all_skipped(self) -> None:
        """All Roman-numeral-labeled pages are dropped, not just the first."""
        texts = [
            make_text_item("Front matter p1.", page_no=1),
            make_text_item("Front matter p2.", page_no=2),
            make_text_item("Body text.", page_no=3),
        ]
        parser = make_parser(texts, page_labels={0: 'i', 1: 'ii', 2: '1'}, skip_front_matter=True)
        chunks = parser._extract_chunks(texts, [])
        assert len(chunks) == 1
        assert "Body text." in chunks[0].text
