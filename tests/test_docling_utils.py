"""Tests for utils.docling_utils helper functions."""

from unittest.mock import MagicMock
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem, DocItemLabel
from utils.docling_utils import (
    is_section_header, is_page_footer, is_page_header, is_footnote,
    is_list_item, is_text_break, is_page_not_text, is_page_text,
    is_too_short, is_text_item, get_next_text,
    get_current_page, should_skip_element
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
        assert clean_text("ﬀect") == "ffect"

    def test_normalizes_left_double_quote(self) -> None:
        assert clean_text("\u201chello\u201d") == '"hello"'

    def test_normalizes_smart_single_quotes(self) -> None:
        assert clean_text("\u2018hello\u2019") == "'hello'"

    def test_normalizes_right_single_quote_possessive(self) -> None:
        assert clean_text("dog\u2019s") == "dog's"

    # def test_removes_soft_hyphen(self):
    #     assert clean_text("explo\u00adration") == "exploration"
    #
    # def test_removes_soft_hyphen_at_word_boundary(self):
    #     assert clean_text("some\u00ad thing") == "some thing"

    def test_preserves_regular_hyphen(self) -> None:
        assert clean_text("well-known") == "well-known"
