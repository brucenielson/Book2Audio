"""Tests for utils.docling_utils helper functions."""

from unittest.mock import MagicMock
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem, DocItemLabel
from utils.docling_utils import (
    is_section_header, is_page_footer, is_page_header, is_footnote,
    is_list_item, is_text_break, is_page_not_text, is_page_text,
    is_too_short, is_text_item, get_next_text,
    get_current_page, should_skip_element,
    compute_single_line_height, compute_median_chars_per_line, is_small_text,
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


# --- is_page_not_text ---

class TestIsPageNotText:
    def test_returns_false_for_text(self) -> None:
        assert is_page_not_text(make_text_item(DocItemLabel.TEXT.value)) is False

    def test_returns_false_for_list_item(self) -> None:
        assert is_page_not_text(make_list_item()) is False

    def test_returns_true_for_page_header(self) -> None:
        assert is_page_not_text(make_text_item(DocItemLabel.PAGE_HEADER.value)) is True

    def test_returns_true_for_none(self) -> None:
        assert is_page_not_text(None) is True


# --- is_page_text ---

class TestIsPageText:
    def test_returns_true_for_text(self) -> None:
        assert is_page_text(make_text_item(DocItemLabel.TEXT.value)) is True

    def test_returns_false_for_page_header(self) -> None:
        assert is_page_text(make_text_item(DocItemLabel.PAGE_HEADER.value)) is False

    def test_returns_false_for_none(self) -> None:
        assert is_page_text(None) is False


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
