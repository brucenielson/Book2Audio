"""Tests for utils.docling_utils helper functions."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem, DocItemLabel
from utils.docling_utils import (
    is_section_header, is_page_footer, is_page_header, is_footnote,
    is_list_item, is_text_break, is_body_text,
    is_too_short, is_text_item, get_next_text,
    get_current_page, should_skip_element,
    compute_single_line_height, compute_median_chars_per_line, is_small_text,
    compute_body_line_height,
    is_single_line,
    get_pdf_page_labels, is_front_matter,
    calibrate_header_top_y, compute_median_page_height,
)
from utils.general_utils import clean_text


# --- Fixtures ---

def make_text_item(label: str, text: str = "Sample text.") -> MagicMock:
    """Create a mock DocItem with the given label and text."""
    item = MagicMock(spec=TextItem)
    item.label = label
    item.text = text
    item.prov = [MagicMock(page_no=1)]
    return item


def make_section_header(text: str = "Chapter 1") -> MagicMock:
    item = MagicMock(spec=SectionHeaderItem)
    item.label = DocItemLabel.SECTION_HEADER.value
    item.text = text
    item.prov = [MagicMock(page_no=1)]
    return item


def make_list_item(text: str = "List item.") -> MagicMock:
    item = MagicMock(spec=ListItem)
    item.label = DocItemLabel.LIST_ITEM.value
    item.text = text
    item.prov = [MagicMock(page_no=1)]
    return item


# --- is_section_header ---

class TestIsSectionHeader:
    def test_returns_true_for_section_header(self) -> None:
        assert is_section_header(make_section_header()) is True

    def test_returns_false_for_non_section_header(self) -> None:
        assert is_section_header(make_text_item(DocItemLabel.TEXT.value)) is False

    def test_returns_false_for_none(self) -> None:
        assert is_section_header(None) is False

    def test_returns_false_for_plain_doc_item(self) -> None:
        item = MagicMock(spec=DocItem)
        assert is_section_header(item) is False


# --- is_page_footer ---

class TestIsPageFooter:
    def test_returns_true_for_page_footer(self) -> None:
        assert is_page_footer(make_text_item(DocItemLabel.PAGE_FOOTER.value)) is True

    def test_returns_false_for_non_footer(self) -> None:
        assert is_page_footer(make_text_item(DocItemLabel.TEXT.value)) is False

    def test_returns_false_for_none(self) -> None:
        assert is_page_footer(None) is False


# --- is_page_header ---

class TestIsPageHeader:
    def test_returns_true_for_page_header(self) -> None:
        assert is_page_header(make_text_item(DocItemLabel.PAGE_HEADER.value)) is True

    def test_returns_false_for_non_header(self) -> None:
        assert is_page_header(make_text_item(DocItemLabel.TEXT.value)) is False

    def test_returns_false_for_none(self) -> None:
        assert is_page_header(None) is False


# --- is_footnote ---

class TestIsFootnote:
    def test_returns_true_for_footnote(self) -> None:
        assert is_footnote(make_text_item(DocItemLabel.FOOTNOTE.value)) is True

    def test_returns_false_for_non_footnote(self) -> None:
        assert is_footnote(make_text_item(DocItemLabel.TEXT.value)) is False

    def test_returns_false_for_none(self) -> None:
        assert is_footnote(None) is False


# --- is_list_item ---

class TestIsListItem:
    def test_returns_true_for_list_item(self) -> None:
        assert is_list_item(make_list_item()) is True

    def test_returns_false_for_non_list_item(self) -> None:
        assert is_list_item(make_text_item(DocItemLabel.TEXT.value)) is False

    def test_returns_false_for_none(self) -> None:
        assert is_list_item(None) is False


# --- is_text_break ---

class TestIsTextBreak:
    def test_returns_true_for_page_header(self) -> None:
        assert is_text_break(make_text_item(DocItemLabel.PAGE_HEADER.value)) is True

    def test_returns_true_for_section_header(self) -> None:
        assert is_text_break(make_section_header()) is True

    def test_returns_true_for_footnote(self) -> None:
        assert is_text_break(make_text_item(DocItemLabel.FOOTNOTE.value)) is True

    def test_returns_false_for_regular_text(self) -> None:
        assert is_text_break(make_text_item(DocItemLabel.TEXT.value)) is False

    def test_returns_false_for_none(self) -> None:
        assert is_text_break(None) is False


# --- is_body_text ---

class TestIsBodyText:
    def test_returns_true_for_text(self) -> None:
        assert is_body_text(make_text_item(DocItemLabel.TEXT.value)) is True

    def test_returns_true_for_list_item(self) -> None:
        assert is_body_text(make_list_item()) is True

    def test_returns_true_for_formula(self) -> None:
        assert is_body_text(make_text_item(DocItemLabel.FORMULA.value)) is True

    def test_returns_false_for_page_header(self) -> None:
        assert is_body_text(make_text_item(DocItemLabel.PAGE_HEADER.value)) is False

    def test_returns_false_for_section_header(self) -> None:
        assert is_body_text(make_text_item(DocItemLabel.SECTION_HEADER.value)) is False

    def test_returns_false_for_footnote(self) -> None:
        assert is_body_text(make_text_item(DocItemLabel.FOOTNOTE.value)) is False

    def test_returns_false_for_none(self) -> None:
        assert is_body_text(None) is False


# --- is_too_short ---

class TestIsTooShort:
    def test_short_text_item(self) -> None:
        item = MagicMock(spec=TextItem)
        item.text = "Hi"
        assert is_too_short(item) is True

    def test_long_text_item(self) -> None:
        item = MagicMock(spec=TextItem)
        item.text = "This is a longer sentence."
        assert is_too_short(item) is False

    def test_non_text_item(self) -> None:
        item = MagicMock(spec=DocItem)
        assert is_too_short(item) is False

    def test_custom_threshold(self) -> None:
        item = MagicMock(spec=TextItem)
        item.text = "Hello"
        assert is_too_short(item, threshold=10) is True


# --- is_text_item ---

class TestIsTextItem:
    def test_regular_text_is_text_item(self) -> None:
        assert is_text_item(make_text_item(DocItemLabel.TEXT.value)) is True

    def test_section_header_is_not_text_item(self) -> None:
        assert is_text_item(make_section_header()) is False

    def test_page_footer_is_not_text_item(self) -> None:
        assert is_text_item(make_text_item(DocItemLabel.PAGE_FOOTER.value)) is False

    def test_page_header_is_not_text_item(self) -> None:
        assert is_text_item(make_text_item(DocItemLabel.PAGE_HEADER.value)) is False

    def test_none_is_not_text_item(self) -> None:
        assert is_text_item(None) is False

    def test_plain_doc_item_is_not_text_item(self) -> None:
        assert is_text_item(MagicMock(spec=DocItem)) is False


# --- get_next_text ---

class TestGetNextText:
    def test_returns_next_text_item(self) -> None:
        items = [
            make_text_item(DocItemLabel.TEXT.value),
            make_text_item(DocItemLabel.TEXT.value),
        ]
        result = get_next_text(items, 0)
        assert result is items[1]

    def test_skips_non_text_items(self) -> None:
        items = [
            make_text_item(DocItemLabel.TEXT.value),
            make_text_item(DocItemLabel.PAGE_HEADER.value),
            make_text_item(DocItemLabel.TEXT.value),
        ]
        result = get_next_text(items, 0)
        assert result is items[2]

    def test_returns_none_at_end(self) -> None:
        items = [make_text_item(DocItemLabel.TEXT.value)]
        assert get_next_text(items, 0) is None

    def test_returns_none_for_empty_list(self) -> None:
        assert get_next_text([], 0) is None


# --- get_current_page ---

class TestGetCurrentPage:
    def test_returns_page_no_when_current_page_is_none(self) -> None:
        item = make_text_item(DocItemLabel.TEXT.value)
        item.prov[0].page_no = 5
        assert get_current_page(item, "", None) == 5

    def test_returns_existing_page_when_paragraph_in_progress(self) -> None:
        item = make_text_item(DocItemLabel.TEXT.value)
        item.prov[0].page_no = 5
        assert get_current_page(item, "some text", 3) == 3

    def test_returns_current_page_for_non_text_item(self) -> None:
        item = MagicMock(spec=DocItem)
        assert get_current_page(item, "", 7) == 7


# --- should_skip_element ---

class TestShouldSkipElement:
    def test_skips_page_footer(self) -> None:
        assert should_skip_element(make_text_item(DocItemLabel.PAGE_FOOTER.value)) is True

    def test_skips_page_header(self) -> None:
        assert should_skip_element(make_text_item(DocItemLabel.PAGE_HEADER.value)) is True

    def test_skips_roman_numeral(self) -> None:
        assert should_skip_element(make_text_item(DocItemLabel.TEXT.value, "XIV")) is False

    def test_does_not_skip_regular_text(self) -> None:
        assert should_skip_element(make_text_item(DocItemLabel.TEXT.value, "Hello world.")) is False

    def test_skips_non_text_item(self) -> None:
        assert should_skip_element(MagicMock(spec=DocItem)) is True


# --- clean_text ---

class TestCleanText:
    def test_strips_whitespace(self) -> None:
        assert clean_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self) -> None:
        assert clean_text("hello   world") == "hello world"

    def test_removes_space_before_period(self) -> None:
        assert clean_text("hello .") == "hello."

    def test_removes_space_before_comma(self) -> None:
        assert clean_text("hello , world") == "hello, world"

    def test_removes_space_before_question_mark(self) -> None:
        assert clean_text("really ?") == "really?"

    def test_removes_space_before_exclamation(self) -> None:
        assert clean_text("wow !") == "wow!"

    def test_removes_space_inside_parentheses(self) -> None:
        assert clean_text("( hello )") == "(hello)"

    def test_fixes_possessive_apostrophe(self) -> None:
        assert clean_text("the dog 's bone") == "the dog's bone"

    def test_strips_trailing_footnote_numbers(self) -> None:
        assert clean_text("Hello world.1", remove_footnotes=True) == "Hello world."

    def test_strips_multiple_trailing_footnote_numbers(self) -> None:
        assert clean_text("Hello world.123", remove_footnotes=True) == "Hello world."

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_normalizes_fi_ligature(self) -> None:
        assert clean_text("ﬁle") == "file"

    def test_normalizes_fl_ligature(self) -> None:
        assert clean_text("ﬂoor") == "floor"

    def test_normalizes_ff_ligature(self) -> None:
        # noinspection SpellCheckingInspection
        assert clean_text("ﬀect") == "ffect"

    def test_normalizes_left_double_quote(self) -> None:
        assert clean_text("\u201chello\u201d") == '"hello"'

    def test_normalizes_smart_single_quotes(self) -> None:
        assert clean_text("\u2018hello\u2019") == "'hello'"

    def test_normalizes_right_single_quote_possessive(self) -> None:
        assert clean_text("dog\u2019s") == "dog's"


    def test_preserves_regular_hyphen(self) -> None:
        assert clean_text("well-known") == "well-known"


# --- Helpers for bbox/charspan mocks ---

def make_text_item_with_bbox(label: str, height: float,
                              charspan_start: int, charspan_end: int) -> MagicMock:
    """Create a mock TextItem with label, bbox height, and charspan attributes."""
    item = MagicMock(spec=TextItem)
    item.label = label
    prov = MagicMock()
    prov.bbox = MagicMock()
    prov.bbox.height = height
    prov.charspan = (charspan_start, charspan_end)
    item.prov = [prov]
    return item


def make_doc_with_texts(items: list) -> MagicMock:
    """Create a mock DoclingDocument with the given texts list."""
    doc = MagicMock(spec=DoclingDocument)
    doc.texts = items
    return doc


# --- compute_single_line_height ---

class TestComputeSingleLineHeight:
    def test_returns_median_of_page_headers(self) -> None:
        items = [
            make_text_item_with_bbox(DocItemLabel.PAGE_HEADER.value, height=12.0, charspan_start=0, charspan_end=10),
            make_text_item_with_bbox(DocItemLabel.PAGE_HEADER.value, height=14.0, charspan_start=0, charspan_end=10),
            make_text_item_with_bbox(DocItemLabel.PAGE_HEADER.value, height=13.0, charspan_start=0, charspan_end=10),
        ]
        doc = make_doc_with_texts(items)
        assert compute_single_line_height(doc) == 13.0

    def test_includes_page_footers(self) -> None:
        items = [
            make_text_item_with_bbox(DocItemLabel.PAGE_FOOTER.value, height=11.0, charspan_start=0, charspan_end=10),
        ]
        doc = make_doc_with_texts(items)
        assert compute_single_line_height(doc) == 11.0

    def test_ignores_body_text(self) -> None:
        items = [
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=50.0, charspan_start=0, charspan_end=200),
        ]
        doc = make_doc_with_texts(items)
        assert compute_single_line_height(doc) == 0.0

    def test_returns_zero_for_empty_doc(self) -> None:
        doc = make_doc_with_texts([])
        assert compute_single_line_height(doc) == 0.0

    def test_skips_item_with_no_prov(self) -> None:
        item = MagicMock(spec=TextItem)
        item.label = DocItemLabel.PAGE_HEADER.value
        item.prov = []
        doc = make_doc_with_texts([item])
        assert compute_single_line_height(doc) == 0.0

    def test_skips_item_with_none_bbox(self) -> None:
        item = MagicMock(spec=TextItem)
        item.label = DocItemLabel.PAGE_HEADER.value
        prov = MagicMock()
        prov.bbox = None
        item.prov = [prov]
        doc = make_doc_with_texts([item])
        assert compute_single_line_height(doc) == 0.0


# --- compute_median_chars_per_line ---

class TestComputeMedianCharsPerLine:
    def test_single_item(self) -> None:
        # single_line_height=10, bbox.height=20 → estimated_lines=2, charspan=100 → 50 chars/line
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=20.0, charspan_start=0, charspan_end=100)
        assert compute_median_chars_per_line([item], single_line_height=10.0) == 50.0

    def test_returns_median_of_multiple_items(self) -> None:
        # chars/line: 50, 100, 75 → sorted: 50, 75, 100 → median = 75
        items = [
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=20.0, charspan_start=0, charspan_end=100),
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=20.0, charspan_start=0, charspan_end=200),
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=20.0, charspan_start=0, charspan_end=150),
        ]
        assert compute_median_chars_per_line(items, single_line_height=10.0) == 75.0

    def test_returns_zero_when_single_line_height_is_zero(self) -> None:
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=20.0, charspan_start=0, charspan_end=100)
        assert compute_median_chars_per_line([item], single_line_height=0.0) == 0.0

    def test_returns_zero_for_empty_list(self) -> None:
        assert compute_median_chars_per_line([], single_line_height=10.0) == 0.0

    def test_skips_item_with_no_prov(self) -> None:
        item = MagicMock(spec=TextItem)
        item.prov = []
        assert compute_median_chars_per_line([item], single_line_height=10.0) == 0.0

    def test_skips_item_with_none_bbox(self) -> None:
        item = MagicMock(spec=TextItem)
        prov = MagicMock()
        prov.bbox = None
        item.prov = [prov]
        assert compute_median_chars_per_line([item], single_line_height=10.0) == 0.0

    def test_skips_item_with_zero_charspan(self) -> None:
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=20.0, charspan_start=5, charspan_end=5)
        assert compute_median_chars_per_line([item], single_line_height=10.0) == 0.0


# --- compute_body_line_height ---

class TestComputeBodyLineHeight:

    def test_returns_75th_percentile_of_single_line_text_items(self) -> None:
        # All three TEXT items, all single-line (height <= 12.0 * 1.3 = 15.6)
        # Sorted heights: [10, 11, 12] → 75th percentile index 3*3//4=2 → 12.0
        items = [
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=100),
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=11.0, charspan_start=0, charspan_end=100),
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=12.0, charspan_start=0, charspan_end=100),
        ]
        assert compute_body_line_height(items, single_line_height=12.0) == 12.0

    def test_75th_percentile_resists_footnote_contamination(self) -> None:
        # 6 items at height 9.0 (footnotes) + 4 items at height 11.0 (body text)
        # Sorted: [9, 9, 9, 9, 9, 9, 11, 11, 11, 11] (10 items)
        # Median index 5 → 9.0  (WRONG — dragged down by footnote-sized items)
        # 75th percentile index 3*10//4=7 → 11.0  (CORRECT — body text height)
        items = (
            [make_text_item_with_bbox(DocItemLabel.TEXT.value, height=9.0, charspan_start=0, charspan_end=50)] * 6
            + [make_text_item_with_bbox(DocItemLabel.TEXT.value, height=11.0, charspan_start=0, charspan_end=100)] * 4
        )
        assert compute_body_line_height(items, single_line_height=12.0) == 11.0

    def test_excludes_multi_line_text_items(self) -> None:
        # single_line_height=12.0; height=10 → single-line (10 <= 15.6) included;
        # height=30 → multi-line (30 > 15.6) excluded
        items = [
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=100),
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=30.0, charspan_start=0, charspan_end=400),
        ]
        assert compute_body_line_height(items, single_line_height=12.0) == 10.0

    def test_ignores_non_text_items(self) -> None:
        # PAGE_HEADER and SECTION_HEADER should be ignored; only TEXT items count
        items = [
            make_text_item_with_bbox(DocItemLabel.PAGE_HEADER.value, height=8.0, charspan_start=0, charspan_end=50),
            make_text_item_with_bbox(DocItemLabel.SECTION_HEADER.value, height=20.0, charspan_start=0, charspan_end=50),
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=100),
        ]
        assert compute_body_line_height(items, single_line_height=12.0) == 10.0

    def test_ignores_footnote_items(self) -> None:
        items = [
            make_text_item_with_bbox(DocItemLabel.FOOTNOTE.value, height=8.0, charspan_start=0, charspan_end=50),
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=100),
        ]
        assert compute_body_line_height(items, single_line_height=12.0) == 10.0

    def test_returns_zero_when_no_text_items(self) -> None:
        items = [
            make_text_item_with_bbox(DocItemLabel.PAGE_HEADER.value, height=10.0, charspan_start=0, charspan_end=50),
        ]
        assert compute_body_line_height(items, single_line_height=12.0) == 0.0

    def test_returns_zero_when_single_line_height_is_zero(self) -> None:
        items = [
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=100),
        ]
        assert compute_body_line_height(items, single_line_height=0.0) == 0.0

    def test_returns_zero_when_list_is_empty(self) -> None:
        assert compute_body_line_height([], single_line_height=10.0) == 0.0

    def test_returns_zero_when_all_text_items_are_multi_line(self) -> None:
        # All heights exceed single_line_height * 1.3 — nothing passes the single-line filter
        items = [
            make_text_item_with_bbox(DocItemLabel.TEXT.value, height=50.0, charspan_start=0, charspan_end=400),
        ]
        assert compute_body_line_height(items, single_line_height=12.0) == 0.0

    def test_skips_item_with_no_prov(self) -> None:
        item = MagicMock(spec=TextItem)
        item.label = DocItemLabel.TEXT.value
        item.prov = []
        assert compute_body_line_height([item], single_line_height=10.0) == 0.0

    def test_skips_item_with_none_bbox(self) -> None:
        item = MagicMock(spec=TextItem)
        item.label = DocItemLabel.TEXT.value
        prov = MagicMock()
        prov.bbox = None
        item.prov = [prov]
        assert compute_body_line_height([item], single_line_height=10.0) == 0.0


# --- is_small_text ---

class TestIsSmallText:
    def test_returns_true_when_chars_per_line_above_threshold(self) -> None:
        # single_line_height=10, bbox.height=10 → estimated_lines=1, charspan=200 → 200 chars/line
        # median=100, threshold=1.25 → 200 > 125 → True
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=200)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=100.0) is True

    def test_returns_false_when_chars_per_line_at_threshold(self) -> None:
        # chars/line = 125; median=100; threshold=1.25 → 125 is NOT > 125
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=125)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=100.0) is False

    def test_returns_false_when_chars_per_line_below_threshold(self) -> None:
        # chars/line = 80; median=100; threshold=1.25 → 80 < 125 → False
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=80)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=100.0) is False

    def test_returns_false_when_single_line_height_is_zero(self) -> None:
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=200)
        assert is_small_text(item, single_line_height=0.0, median_chars_per_line=100.0) is False

    def test_returns_false_when_median_is_zero(self) -> None:
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=200)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=0.0) is False

    def test_returns_false_when_no_prov(self) -> None:
        item = MagicMock(spec=TextItem)
        item.prov = []
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=100.0) is False

    def test_returns_false_when_bbox_is_none(self) -> None:
        item = MagicMock(spec=TextItem)
        prov = MagicMock()
        prov.bbox = None
        item.prov = [prov]
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=100.0) is False

    def test_returns_false_when_charspan_zero_length(self) -> None:
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=5, charspan_end=5)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=100.0) is False

    def test_custom_threshold(self) -> None:
        # chars/line=200, median=100, threshold=2.5 → 200 is NOT > 250 → False
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=200)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=100.0, threshold=2.5) is False

    # --- body_line_height path ---

    def test_body_line_height_fires_for_short_item(self) -> None:
        # Short item (10 chars): chars-per-line = 10/1 = 10, well below median → not small via chars-per-line.
        # But bbox.height=8.0 < body_line_height=10.0 * body_threshold=0.85 → 8.5 → True via body path.
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=8.0, charspan_start=0, charspan_end=10)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=80.0,
                             body_line_height=10.0) is True

    def test_body_line_height_does_not_fire_when_height_at_threshold(self) -> None:
        # bbox.height=8.5 is NOT < 10.0 * 0.85 = 8.5 (strict less-than) → body path fails.
        # Chars-per-line: 10/1=10 << 80*1.25 → False overall.
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=8.5, charspan_start=0, charspan_end=10)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=80.0,
                             body_line_height=10.0) is False

    def test_body_line_height_zero_falls_back_to_chars_per_line(self) -> None:
        # body_line_height=0 → skip body check.
        # Chars-per-line: 200/1=200 > 80*1.25=100 → True via chars-per-line.
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=10.0, charspan_start=0, charspan_end=200)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=80.0,
                             body_line_height=0.0) is True

    def test_custom_body_threshold(self) -> None:
        # bbox.height=8.5 NOT < 10.0 * 0.85=8.5 with default threshold → False.
        # With body_threshold=0.9: 8.5 < 10.0*0.9=9.0 → True.
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=8.5, charspan_start=0, charspan_end=10)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=80.0,
                             body_line_height=10.0, body_threshold=0.9) is True

    def test_body_line_height_path_takes_priority_over_chars_per_line(self) -> None:
        # Long item that would pass chars-per-line, but body check fires first.
        # Both should give True — verifying body path fires first (for correctness, same result).
        item = make_text_item_with_bbox(DocItemLabel.TEXT.value, height=8.0, charspan_start=0, charspan_end=200)
        assert is_small_text(item, single_line_height=10.0, median_chars_per_line=80.0,
                             body_line_height=10.0) is True


# --- TestIsSingleLine ---

class TestIsSingleLine:

    def test_returns_true_when_height_equals_single_line(self) -> None:
        item = make_text_item_with_bbox(DocItemLabel.SECTION_HEADER.value, height=10.0, charspan_start=0, charspan_end=20)
        assert is_single_line(item, single_line_height=10.0) is True

    def test_returns_true_when_height_within_tolerance(self) -> None:
        # height=14.0, single_line=10.0, tolerance=1.5 → 14 <= 15 → True
        item = make_text_item_with_bbox(DocItemLabel.SECTION_HEADER.value, height=12.0, charspan_start=0, charspan_end=20)
        assert is_single_line(item, single_line_height=10.0) is True

    def test_returns_false_when_height_exceeds_tolerance(self) -> None:
        # height=16.0, single_line=10.0, tolerance=1.5 → 16 > 15 → False
        item = make_text_item_with_bbox(DocItemLabel.SECTION_HEADER.value, height=16.0, charspan_start=0, charspan_end=20)
        assert is_single_line(item, single_line_height=10.0) is False

    def test_returns_false_when_single_line_height_is_zero(self) -> None:
        item = make_text_item_with_bbox(DocItemLabel.SECTION_HEADER.value, height=10.0, charspan_start=0, charspan_end=20)
        assert is_single_line(item, single_line_height=0.0) is False

    def test_returns_false_when_prov_is_empty(self) -> None:
        item = MagicMock(spec=TextItem)
        item.prov = []
        assert is_single_line(item, single_line_height=10.0) is False

    def test_returns_false_when_bbox_is_none(self) -> None:
        item = MagicMock(spec=TextItem)
        prov = MagicMock()
        prov.bbox = None
        item.prov = [prov]
        assert is_single_line(item, single_line_height=10.0) is False

    def test_custom_tolerance(self) -> None:
        # height=12.0, single_line=10.0, tolerance=1.1 → 12 > 11 → False
        item = make_text_item_with_bbox(DocItemLabel.SECTION_HEADER.value, height=12.0, charspan_start=0, charspan_end=20)
        assert is_single_line(item, single_line_height=10.0, tolerance=1.1) is False

    def test_exactly_at_tolerance_boundary(self) -> None:
        # height=15.0, single_line=10.0, tolerance=1.5 → 15 <= 15 → True
        item = make_text_item_with_bbox(DocItemLabel.SECTION_HEADER.value, height=13.0, charspan_start=0, charspan_end=20)
        assert is_single_line(item, single_line_height=10.0) is True


# --- get_pdf_page_labels ---

class TestGetPdfPageLabels:
    """Tests for get_pdf_page_labels(path) -> dict[int, str].

    Page labels are the logical page numbers printed in the book (e.g. Roman
    numerals 'i', 'ii'... 'xl' for front matter, then '1', '2'... for body).
    Physical page indices are sequential from 0 regardless of what's printed.
    """

    def _make_mock_pdf(self, labels: list[str]) -> MagicMock:
        """Build a mock PdfDocument that returns the given labels by index."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=len(labels))
        mock_doc.get_page_label = MagicMock(side_effect=lambda i: labels[i])
        return mock_doc

    def test_returns_mapping_of_index_to_label(self) -> None:
        """Basic case: physical indices map to their printed labels."""
        labels = ['i', 'ii', 'iii', '1', '2', '3']
        mock_doc = self._make_mock_pdf(labels)
        with patch('utils.docling_utils.pypdfium2.PdfDocument', return_value=mock_doc):
            result = get_pdf_page_labels(Path('dummy.pdf'))
        assert result == {0: 'i', 1: 'ii', 2: 'iii', 3: '1', 4: '2', 5: '3'}

    def test_all_arabic_no_front_matter(self) -> None:
        """PDF with no front matter — all labels are Arabic numerals."""
        labels = ['1', '2', '3', '4']
        mock_doc = self._make_mock_pdf(labels)
        with patch('utils.docling_utils.pypdfium2.PdfDocument', return_value=mock_doc):
            result = get_pdf_page_labels(Path('dummy.pdf'))
        assert result == {0: '1', 1: '2', 2: '3', 3: '4'}

    def test_long_roman_numeral_front_matter(self) -> None:
        """40 pages of Roman numeral front matter (xl) before body text."""
        roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
                 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
                 'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi', 'xxvii', 'xxviii', 'xxix', 'xxx',
                 'xxxi', 'xxxii', 'xxxiii', 'xxxiv', 'xxxv', 'xxxvi', 'xxxvii', 'xxxviii', 'xxxix', 'xl']
        labels = roman + ['1', '2', '3']
        mock_doc = self._make_mock_pdf(labels)
        with patch('utils.docling_utils.pypdfium2.PdfDocument', return_value=mock_doc):
            result = get_pdf_page_labels(Path('dummy.pdf'))
        assert result[0] == 'i'
        assert result[39] == 'xl'
        assert result[40] == '1'
        assert result[42] == '3'


# --- Helpers for calibrate_header_top_y / compute_median_page_height ---

def make_page_header_item_with_t(bbox_t: float, page_no: int = 1) -> MagicMock:
    """Create a mock PAGE_HEADER TextItem with a specific bbox.t value."""
    item = MagicMock(spec=TextItem)
    item.label = DocItemLabel.PAGE_HEADER
    prov = MagicMock()
    prov.bbox = MagicMock()
    prov.bbox.t = bbox_t
    item.prov = [prov]
    return item


def make_doc_with_pages(heights: list[float]) -> MagicMock:
    """Create a mock DoclingDocument whose pages dict contains pages with the given heights."""
    doc = MagicMock(spec=DoclingDocument)
    doc.texts = []
    pages = {}
    for i, h in enumerate(heights, start=1):
        page = MagicMock()
        page.size = MagicMock()
        page.size.height = h
        pages[i] = page
    doc.pages = pages
    return doc


# --- calibrate_header_top_y ---

class TestCalibrateHeaderTopY:
    def test_no_page_headers_returns_none(self) -> None:
        """No PAGE_HEADER items in doc → returns None."""
        body = make_text_item(DocItemLabel.TEXT.value)
        doc = make_doc_with_texts([body])
        assert calibrate_header_top_y(doc) is None

    def test_single_page_header_returns_its_t(self) -> None:
        """A single PAGE_HEADER → its bbox.t is returned."""
        header = make_page_header_item_with_t(bbox_t=20.0)
        doc = make_doc_with_texts([header])
        assert calibrate_header_top_y(doc) == pytest.approx(20.0)

    def test_page_header_after_body_text_still_counted(self) -> None:
        """PAGE_HEADER that appears after body text in doc.texts is still used.
        Docling does not guarantee page headers are listed first — only the label matters."""
        body = make_text_item(DocItemLabel.TEXT.value)
        header = make_page_header_item_with_t(bbox_t=20.0)
        doc = make_doc_with_texts([body, header])  # header comes after body
        assert calibrate_header_top_y(doc) == pytest.approx(20.0)

    def test_multiple_page_headers_returns_median(self) -> None:
        """Multiple PAGE_HEADERs → returns median bbox.t."""
        h1 = make_page_header_item_with_t(bbox_t=20.0)
        h2 = make_page_header_item_with_t(bbox_t=22.0)
        h3 = make_page_header_item_with_t(bbox_t=18.0)
        doc = make_doc_with_texts([h1, h2, h3])
        assert calibrate_header_top_y(doc) == pytest.approx(20.0)  # median of [18, 20, 22]

    def test_page_header_with_none_bbox_is_skipped(self) -> None:
        """A PAGE_HEADER whose bbox is None is excluded from the median."""
        good = make_page_header_item_with_t(bbox_t=20.0)
        bad = MagicMock(spec=TextItem)
        bad.label = DocItemLabel.PAGE_HEADER
        prov = MagicMock()
        prov.bbox = None
        bad.prov = [prov]
        doc = make_doc_with_texts([good, bad])
        assert calibrate_header_top_y(doc) == pytest.approx(20.0)

    def test_page_header_with_no_prov_is_skipped(self) -> None:
        """A PAGE_HEADER with no prov is excluded."""
        good = make_page_header_item_with_t(bbox_t=20.0)
        bad = MagicMock(spec=TextItem)
        bad.label = DocItemLabel.PAGE_HEADER
        bad.prov = []
        doc = make_doc_with_texts([good, bad])
        assert calibrate_header_top_y(doc) == pytest.approx(20.0)


# --- compute_median_page_height ---

class TestComputeMedianPageHeight:
    def test_no_pages_returns_zero(self) -> None:
        """Doc with no pages data → 0.0."""
        doc = MagicMock(spec=DoclingDocument)
        doc.pages = {}
        assert compute_median_page_height(doc) == 0.0

    def test_single_page_returns_its_height(self) -> None:
        """A single page → its height is returned."""
        doc = make_doc_with_pages([700.0])
        assert compute_median_page_height(doc) == pytest.approx(700.0)

    def test_multiple_pages_returns_median(self) -> None:
        """Three pages → median height returned."""
        doc = make_doc_with_pages([600.0, 800.0, 700.0])
        assert compute_median_page_height(doc) == pytest.approx(700.0)  # median of [600, 700, 800]

    def test_page_with_none_size_is_skipped(self) -> None:
        """Page whose size is None is excluded from the median."""
        doc = make_doc_with_pages([700.0])
        bad_page = MagicMock()
        bad_page.size = None
        doc.pages[99] = bad_page
        assert compute_median_page_height(doc) == pytest.approx(700.0)

    def test_no_pages_attribute_returns_zero(self) -> None:
        """Doc with no 'pages' attribute at all → 0.0."""
        doc = MagicMock(spec=DoclingDocument)
        del doc.pages
        assert compute_median_page_height(doc) == 0.0


# --- is_front_matter ---

class TestIsFrontMatter:
    """Tests for is_front_matter(label) -> bool."""

    @pytest.mark.parametrize("label,expected", [
        ('i',      True),
        ('iv',     True),
        ('xl',     True),
        ('xii',    True),
        ('1',      False),
        ('42',     False),
        ('368',    False),
        ('',       False),
    ])
    def test_label_classification(self, label: str, expected: bool) -> None:
        assert is_front_matter(label) is expected
