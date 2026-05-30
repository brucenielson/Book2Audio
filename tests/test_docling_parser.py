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
                skip_front_matter: bool = False,
                skip_index: bool = False,
                page_height: float | None = None) -> DoclingParser:
    """Create a DoclingParser with a mocked DoclingDocument.

    Pass page_height to give compute_median_page_height a non-zero result,
    which enables the lower-half positional checks in _is_footnote.
    """
    doc = MagicMock(spec=DoclingDocument)
    doc.name = "test_doc"
    doc.texts = texts
    if page_height is not None:
        mock_page = MagicMock()
        mock_page.size = MagicMock()
        mock_page.size.height = page_height
        doc.pages = {1: mock_page}
    else:
        doc.pages = {}
    return DoclingParser(source=doc, meta_data=meta_data or {}, min_paragraph_size=min_paragraph_size,
                         start_page=start_page, end_page=end_page, include_footnotes=include_notes,
                         llm_cleaner=cleaner, min_footnote_chars=min_footnote_chars,
                         page_labels=page_labels, skip_front_matter=skip_front_matter,
                         skip_index=skip_index)


def make_ctx(
    prev_text_candidate: bool = False,
    text_seen_this_page: bool = False,
    found_note_this_page: bool = False,
    single_line_height: float = 10.0,
    median_chars_per_line: float = 50.0,
    header_top_y: float | None = None,
    median_page_height: float = 0.0,
    body_line_height: float = 0.0,
    in_notes_section: bool = False,
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
        body_line_height=body_line_height,
        in_notes_section=in_notes_section,
    )


def make_sized_text_item(text: str, page_no: int = 1,
                         charspan_length: int = 10,
                         bbox_height: float = 10.0) -> MagicMock:
    """Create a TEXT item with configurable charspan and bbox height for H2 testing."""
    item = make_doc_item(TextItem, DocItemLabel.TEXT.value, text, page_no)
    item.prov[0].charspan = (0, charspan_length)
    item.prov[0].bbox.height = bbox_height
    return item


def make_sized_list_item(text: str, charspan_length: int = 200,
                          bbox_height: float = 10.0) -> MagicMock:
    """Create a LIST_ITEM with configurable charspan and bbox height."""
    item = make_doc_item(TextItem, DocItemLabel.LIST_ITEM.value, text)
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
        """LIST_ITEM with digit-start is not a footnote when text_seen_this_page is False."""
        item = make_doc_item(TextItem, DocItemLabel.LIST_ITEM.value, "1 list entry")
        parser = make_parser([])
        assert parser._is_footnote(item, make_ctx(prev_text_candidate=True)) is False

    # --- Digit-start LIST_ITEM as first footnote on page ---
    # Docling often labels bottom-of-page footnotes as LIST_ITEM when they use
    # a "1." or "1 " style marker. Detect via small text + lower-half position.

    def test_digit_list_item_small_lower_half_returns_true(self) -> None:
        """Small digit-start LIST_ITEM in lower half of page with body text seen is a footnote."""
        item = make_sized_list_item("1. See Smith v. Jones, 42 U.S. 100 (1900).")
        item.prov[0].bbox.t = 30.0  # lower half of 100-height page
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=50.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is True

    def test_digit_list_item_not_small_returns_false(self) -> None:
        """A body-sized LIST_ITEM starting with a digit is not a footnote."""
        item = make_sized_list_item("1. See Smith v. Jones, 42 U.S. 100 (1900).")
        item.prov[0].bbox.t = 30.0
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=200.0,  # high median → not small
                       median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is False

    def test_digit_list_item_upper_half_returns_false(self) -> None:
        """A small digit-start LIST_ITEM in the upper half of the page is not a footnote."""
        item = make_sized_list_item("1. See Smith v. Jones, 42 U.S. 100 (1900).")
        item.prov[0].bbox.t = 70.0  # upper half of 100-height page
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=50.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is False

    def test_digit_list_item_no_text_seen_returns_false(self) -> None:
        """Without body text seen on the page, a digit-start LIST_ITEM is not a footnote."""
        item = make_sized_list_item("1. See Smith v. Jones, 42 U.S. 100 (1900).")
        item.prov[0].bbox.t = 30.0
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=False, single_line_height=5.0,
                       median_chars_per_line=50.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is False

    def test_letter_start_list_item_returns_false(self) -> None:
        """A small LIST_ITEM starting with a letter in the lower half is not a footnote."""
        item = make_sized_list_item("See Smith v. Jones, 42 U.S. 100 (1900).")
        item.prov[0].bbox.t = 30.0
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=50.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is False

    def test_empty_text_returns_false(self) -> None:
        """Empty string is falsy — guard bails before any heuristic is checked."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item(""), make_ctx()) is False

    def test_text_starting_with_space_returns_false(self) -> None:
        """A leading space before a digit is not a digit-start — guard must not pass."""
        parser = make_parser([])
        ctx = make_ctx(prev_text_candidate=True)
        assert parser._is_footnote(make_text_item(" 1 Leading space."), ctx) is False

    # --- H1: digit-start TEXT following mid-sentence body text, in lower half of page ---

    def test_h1_digit_alpha_after_mid_sentence_lower_half_returns_true(self) -> None:
        """H1 fires when item follows mid-sentence body text and is in the lower half of page.
        Font size is irrelevant — the positional signal is the discriminator."""
        item = make_sized_text_item("1 This is an unlabelled footnote.",
                                    charspan_length=33, bbox_height=10.0)
        item.prov[0].bbox.t = 30.0  # lower half of 100-height page
        parser = make_parser([])
        ctx = make_ctx(prev_text_candidate=True, text_seen_this_page=True,
                       single_line_height=10.0, median_chars_per_line=50.0,
                       body_line_height=10.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is True

    def test_h1_large_text_lower_half_fires(self) -> None:
        """H1 fires for large-text footnotes in the lower half — no small-text gate.
        '8(See Popper's...' is a real-world example: bbox_height~20, clearly in lower half."""
        item = make_sized_text_item("8(See Popper's Logic of Scientific Discovery.)",
                                    charspan_length=46, bbox_height=20.0)
        item.prov[0].bbox.t = 30.0  # lower half of 100-height page
        parser = make_parser([])
        ctx = make_ctx(prev_text_candidate=True, text_seen_this_page=True,
                       single_line_height=10.0, median_chars_per_line=50.0,
                       body_line_height=10.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is True

    def test_h1_upper_half_does_not_fire(self) -> None:
        """H1 must not fire when the item is in the upper half of the page.
        Digit-start items after a mid-sentence paragraph but near the top are not footnotes."""
        item = make_sized_text_item("1 This is an unlabelled footnote.",
                                    charspan_length=33, bbox_height=10.0)
        item.prov[0].bbox.t = 80.0  # upper half of 100-height page
        parser = make_parser([])
        ctx = make_ctx(prev_text_candidate=True, text_seen_this_page=True,
                       single_line_height=10.0, median_chars_per_line=50.0,
                       body_line_height=10.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is False

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
        """Digit-start item in small font in lower half of page → footnote."""
        text = "1" + "a" * 99   # len=100, digit-start, has alpha
        item = make_sized_text_item(text, charspan_length=200, bbox_height=10.0)
        item.prov[0].bbox.t = 30.0   # lower half of 100-height page
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=50.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is True

    def test_h2_fires_without_alpha(self) -> None:
        """Digit-only item in small font in lower half — no alpha required."""
        text = "1" + "0" * 99   # len=100, digit-start, no alpha
        item = make_sized_text_item(text, charspan_length=200, bbox_height=10.0)
        item.prov[0].bbox.t = 30.0   # lower half of 100-height page
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=50.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is True

    def test_h2_upper_half_digit_start_no_other_signal_returns_false(self) -> None:
        """A digit-start item in small font in the upper half of the page without
        H4 or H1 signals is not a footnote — footnotes live at the bottom."""
        text = "1 short"
        item = make_sized_text_item(text, charspan_length=200, bbox_height=10.0)
        item.prov[0].bbox.t = 70.0   # upper half of 100-height page
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(text_seen_this_page=True, single_line_height=5.0,
                       median_chars_per_line=50.0, median_page_height=100.0)
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

    @pytest.mark.parametrize("text", [
        "2 Continuation of a footnote.",
        "2",
        "in which zeros are followed by ones.",
    ])
    def test_h3_after_note(self, text: str) -> None:
        """Once a footnote is seen on the page, all subsequent TEXT items are footnote
        continuations — digit-start, bare number, and non-digit-start alike."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item(text), make_ctx(found_note_this_page=True)) is True

    @pytest.mark.parametrize("text", [
        "2 Some text.",
        "in which zeros are followed by ones.",
    ])
    def test_h3_without_prior_note(self, text: str) -> None:
        """Without found_note_this_page, H3 must not fire for any text pattern."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item(text), make_ctx(found_note_this_page=False)) is False

    def test_h3_formula_after_note_returns_true(self) -> None:
        """A FORMULA item after a footnote on the page is swept up by H3 propagation."""
        item = make_doc_item(TextItem, DocItemLabel.FORMULA.value, "p(x,z) = p(x)")
        parser = make_parser([])
        assert parser._is_footnote(item, make_ctx(found_note_this_page=True)) is True

    def test_h3_formula_without_prior_note_returns_false(self) -> None:
        """H3 formula propagation must not fire when no footnote has been seen yet."""
        item = make_doc_item(TextItem, DocItemLabel.FORMULA.value, "p(x,z) = p(x)")
        parser = make_parser([])
        assert parser._is_footnote(item, make_ctx(found_note_this_page=False)) is False

    def test_h3_list_item_after_note_returns_true(self) -> None:
        """A LIST_ITEM after a footnote on the page is swept up by H3 propagation —
        list items in the footnote section (e.g. numbered conditions) are footnote content."""
        item = make_doc_item(TextItem, DocItemLabel.LIST_ITEM.value,
                             "(1) If p(x,z) >= p(x) and p(y,z) > p(y) then p(x,z) < p(y,z)")
        parser = make_parser([])
        assert parser._is_footnote(item, make_ctx(found_note_this_page=True)) is True

    def test_h3_list_item_without_prior_note_returns_false(self) -> None:
        """H3 list_item propagation must not fire when no footnote has been seen yet."""
        item = make_doc_item(TextItem, DocItemLabel.LIST_ITEM.value,
                             "(1) If p(x,z) >= p(x) and p(y,z) > p(y) then p(x,z) < p(y,z)")
        parser = make_parser([])
        assert parser._is_footnote(item, make_ctx(found_note_this_page=False)) is False

    # --- Endnote path: in_notes_section flag ---

    def test_endnote_path_digit_alpha_returns_true(self) -> None:
        """In the notes section, a digit-start TEXT item with alpha is an endnote —
        regardless of font size, page position, or whether body text has been seen."""
        parser = make_parser([])
        item = make_text_item("9 Manning asserts that what makes an illegal seizure...")
        assert parser._is_footnote(item, make_ctx(in_notes_section=True)) is True

    def test_endnote_path_digit_only_returns_false(self) -> None:
        """In the notes section, a digit-only item (no alpha) is not an endnote."""
        parser = make_parser([])
        item = make_text_item("9")
        assert parser._is_footnote(item, make_ctx(in_notes_section=True)) is False

    def test_endnote_path_not_active_outside_notes_section(self) -> None:
        """The endnote path must not fire when in_notes_section is False."""
        parser = make_parser([])
        item = make_text_item("9 Manning asserts that what makes an illegal seizure...")
        assert parser._is_footnote(item, make_ctx(in_notes_section=False)) is False

    def test_endnote_path_non_digit_start_not_caught(self) -> None:
        """A non-digit-start item is not caught by the endnote path alone —
        H3 propagation handles continuations once the first endnote is found."""
        parser = make_parser([])
        item = make_text_item("restrict enquiry, it cannot induce a specific belief.")
        assert parser._is_footnote(item, make_ctx(in_notes_section=True,
                                                   found_note_this_page=False)) is False

    # --- No heuristic fires ---

    def test_digit_only_text_all_ctx_false_returns_false(self) -> None:
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("42"), make_ctx()) is False

    def test_digit_alpha_text_all_ctx_false_returns_false(self) -> None:
        """Has alpha and digit-start but no context conditions met — must return False."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item("1 Some text."), make_ctx()) is False

    # --- H4: digit(s) immediately followed by uppercase letter ---

    @pytest.mark.parametrize("text, charspan_length, bbox_height", [
        ("3See my Poverty of Historicism.", 31,  8.0),  # single digit + uppercase, small text
        ("14Cf. the earlier discussion.",   29,  8.0),  # two digits + uppercase, small text
        ("3See my Poverty of Historicism.", 31, 10.0),  # normal sized — small-text gate not applied
    ])
    def test_h4_fires(self, text: str, charspan_length: int, bbox_height: float) -> None:
        """H4 fires for digit+uppercase patterns regardless of font size."""
        item = make_sized_text_item(text, charspan_length=charspan_length, bbox_height=bbox_height)
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, single_line_height=10.0,
                       median_chars_per_line=50.0, body_line_height=10.0)
        assert parser._is_footnote(item, ctx) is True

    @pytest.mark.parametrize("bbox_height", [8.0, 10.0])
    def test_h4_requires_text_seen(self, bbox_height: float) -> None:
        """H4 must not fire before body text has been seen, regardless of font size."""
        item = make_sized_text_item("3See my Poverty of Historicism.",
                                    charspan_length=31, bbox_height=bbox_height)
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=False, single_line_height=10.0,
                       median_chars_per_line=50.0, body_line_height=10.0)
        assert parser._is_footnote(item, ctx) is False

    @pytest.mark.parametrize("text", [
        "183See something.",  # three digits before letter — H4 only matches 1-2 digits
        "1st place goes to",  # lowercase after digit — avoids ordinals
        "3 See my text.",     # space between digit and letter — not the H4 pattern
    ])
    def test_h4_pattern_not_caught(self, text: str) -> None:
        """Edge cases that look like H4 but must not trigger it."""
        parser = make_parser([])
        assert parser._is_footnote(make_text_item(text), make_ctx()) is False

    # --- H4 extended: digit immediately followed by punctuation (bracket, paren, quote) ---

    @pytest.mark.parametrize("text, charspan_length", [
        ("3[See The Open Society, vol. ii.]",              33),  # digit + open bracket
        ("8(See Popper's Logic of Scientific Discovery.)", 46),  # digit + open paren
        ("13'fhat is to say, the refutation.",             34),  # digit + apostrophe
    ])
    def test_h4_extended_fires(self, text: str, charspan_length: int) -> None:
        """Digit followed by punctuation (bracket, paren, apostrophe) triggers H4."""
        item = make_sized_text_item(text, charspan_length=charspan_length, bbox_height=10.0)
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, single_line_height=10.0,
                       median_chars_per_line=50.0, body_line_height=10.0)
        assert parser._is_footnote(item, ctx) is True

    def test_h4_bracket_requires_text_seen(self) -> None:
        """digit + bracket H4 still requires body text seen first."""
        item = make_sized_text_item("3[See The Open Society, vol. ii.]",
                                    charspan_length=33, bbox_height=10.0)
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=False, single_line_height=10.0,
                       median_chars_per_line=50.0, body_line_height=10.0)
        assert parser._is_footnote(item, ctx) is False

    def test_h4_fires_on_mislabeled_section_header(self) -> None:
        """Docling sometimes mislabels a footnote as a SECTION_HEADER.
        H4 must fire for digit+uppercase items regardless of whether Docling
        labeled them TEXT or SECTION_HEADER."""
        item = make_section_header("5To make all this quite clear we write")
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True)
        assert parser._is_footnote(item, ctx) is True

    # --- H2 body_line_height path: detects short digit-start footnotes ---

    def test_h2_body_line_height_fires_for_short_item(self) -> None:
        """Short digit-start item detected via body_line_height in the lower half of page.
        bbox.height=8.0 < body_line_height=10.0 * 0.85=8.5 → small text → lower half → True."""
        text = "1 short"
        item = make_sized_text_item(text, charspan_length=len(text), bbox_height=8.0)
        item.prov[0].bbox.t = 30.0   # lower half of 100-height page
        parser = make_parser([], min_footnote_chars=100)
        ctx = make_ctx(text_seen_this_page=True, single_line_height=10.0,
                       median_chars_per_line=80.0, body_line_height=10.0,
                       median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is True

    def test_numbered_list_returns_false(self) -> None:
        """A digit followed by period and space is a numbered list item, not a footnote,
        even when all other conditions (small text, body seen, lower half) are met."""
        text = "1. A numbered proposition about things."
        item = make_sized_text_item(text, charspan_length=len(text), bbox_height=8.0)
        item.prov[0].bbox.t = 30.0   # lower half
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, single_line_height=10.0,
                       median_chars_per_line=50.0, body_line_height=10.0,
                       median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is False

    def test_lower_half_digit_start_small_text_returns_true(self) -> None:
        """A digit-start item in small text in the lower half of the page is a footnote
        even without H4 or H1 signals."""
        text = "1 Some footnote text."
        item = make_sized_text_item(text, charspan_length=len(text), bbox_height=8.0)
        item.prov[0].bbox.t = 30.0   # lower half of 100-height page
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, single_line_height=10.0,
                       median_chars_per_line=50.0, body_line_height=10.0,
                       median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is True

    def test_h2_body_line_height_does_not_fire_when_normal_sized(self) -> None:
        """H2 must not fire when body_line_height is set but item is normal font size."""
        text = "1 short"  # len=7, below min_footnote_chars=100
        item = make_sized_text_item(text, charspan_length=len(text), bbox_height=10.0)
        parser = make_parser([], min_footnote_chars=100)
        # bbox.height=10.0 is NOT < 10.0*0.85=8.5 → body check fails; also too short for chars-per-line
        ctx = make_ctx(text_seen_this_page=True, single_line_height=10.0,
                       median_chars_per_line=80.0, body_line_height=10.0)
        assert parser._is_footnote(item, ctx) is False

    def test_ocr_mangled_first_char_small_text_lower_half_returns_true(self) -> None:
        """OCR can substitute a digit with a symbol (e.g. '&' for '6'), making a
        first.isdigit() guard reject a genuine footnote.
        Small text + body text seen + lower half of page must be sufficient."""
        text = "&a['n( Added 1982) It is possible to make (G) and (H) even more nearly similar"
        item = make_sized_text_item(text, charspan_length=len(text), bbox_height=8.0)
        item.prov[0].bbox.t = 30.0  # lower half of 100-height page
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, single_line_height=10.0,
                       median_chars_per_line=50.0, body_line_height=10.0,
                       median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is True

    def test_ocr_mangled_first_char_upper_half_returns_false(self) -> None:
        """Non-digit-start small text in the upper half of the page without other
        signals must not be classified as a footnote — footnotes live at the bottom."""
        text = "&a['n( Added 1982) It is possible to make (G) and (H) even more nearly similar"
        item = make_sized_text_item(text, charspan_length=len(text), bbox_height=8.0)
        item.prov[0].bbox.t = 70.0  # upper half of 100-height page
        parser = make_parser([])
        ctx = make_ctx(text_seen_this_page=True, prev_text_candidate=False,
                       single_line_height=10.0, median_chars_per_line=50.0,
                       body_line_height=10.0, median_page_height=100.0)
        assert parser._is_footnote(item, ctx) is False


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
        parser = make_parser(texts)
        docs, _ = parser.run()
        assert len(docs) == 1
        assert "Body text." in docs[0]

    def test_notes_excluded_when_include_notes_false(self) -> None:
        texts = [make_footnote("Footnote text.")]
        parser = make_parser(texts, include_notes=False)
        docs, _ = parser.run()
        assert all("Footnote text." not in d for d in docs)

    def test_notes_included_when_include_notes_true(self) -> None:
        texts = [make_footnote("Footnote text.")]
        parser = make_parser(texts, include_notes=True)
        docs, _ = parser.run()
        assert any("Footnote text." in d for d in docs)

    def test_page_range_filters_out_of_range_items(self) -> None:
        texts = [
            make_text_item("Page 2 text.", page_no=2),
            make_text_item("Page 5 text.", page_no=5),
        ]
        parser = make_parser(texts, start_page=5, end_page=10)
        docs, _ = parser.run()
        assert len(docs) == 1
        assert "Page 5 text." in docs[0]

    @pytest.mark.parametrize("key", ["page_#", "physical_page_#"])
    def test_chunk_meta_page_number(self, key: str) -> None:
        texts = [make_text_item("Text.", page_no=42)]
        parser = make_parser(texts)
        _, meta = parser.run()
        assert meta[0][key] == '42'

    def test_chunk_label_matches_item_label(self) -> None:
        texts = [make_text_item("Regular body text here.")]
        parser = make_parser(texts)
        chunks = parser._get_processed_texts()
        body = [c for c in chunks if not c.is_page_header and not c.is_too_short and not c.is_footnote]
        assert body[0].label == DocItemLabel.TEXT




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
        chunks = parser._get_processed_texts()
        regular = [c for c in chunks if not c.is_footnote and not c.is_page_header and not c.is_too_short]
        notes = [c for c in chunks if c.is_footnote]
        assert len(regular) == 1
        assert len(notes) == 1

    def test_skips_too_short_items(self) -> None:
        texts = [
            make_text_item("Hi"),
            make_text_item("This is a longer sentence."),
        ]
        parser = make_parser(texts)
        chunks = parser._get_processed_texts()
        assert len(chunks) == 2
        assert chunks[0].is_too_short
        assert not chunks[1].is_too_short

    def test_document_order_preserved(self) -> None:
        texts = [
            make_footnote("Footnote."),
            make_text_item("Regular text."),
        ]
        parser = make_parser(texts)
        chunks = parser._get_processed_texts()
        assert chunks[0].text == "Footnote."
        assert chunks[1].text == "Regular text."

    def test_empty_document(self) -> None:
        parser = make_parser([])
        chunks = parser._get_processed_texts()
        assert chunks == []


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
        # Preceding text is long and doesn't end with punctuation → footnote heuristic fires.
        # Page footer provides single_line_height; footnote item is smaller than body text.
        preceding = "A" * 100
        footnote_item = make_sized_text_item(
            "1 This is an unlabelled footnote reference.",
            charspan_length=43, bbox_height=8.0,
        )
        texts = [
            make_text_item(preceding),
            footnote_item,
            make_page_footer("1", page_no=1),   # gives single_line_height=10.0
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

    def test_docling_out_of_order_emission_body_text_preserved(self) -> None:
        """When Docling emits a footnote before body text on the same page (i.e. the
        footnote appears earlier in doc.texts even though it is physically at the
        bottom of the page), the body text must not be swept by H3.

        Regression: Realism and the Aim of Science, p. 252 (physical p. 293).
        Footnotes 6, 7, 8 are emitted by Docling before the body text paragraphs.
        With the unconditional H3, the first footnote sets found_note_this_page=True
        and all subsequent TEXT items — including body text — are swept as footnotes.

        Fix: sort each page's items by bbox.t descending before classification so
        body text (high bbox.t) is always processed before footnotes (low bbox.t),
        regardless of Docling's emission order.
        """
        # Page 1: long body text ending mid-sentence → sets prev_text_candidate=True
        body_p1 = make_sized_text_item("A" * 150, charspan_length=150, bbox_height=10.0,
                                        page_no=1)
        body_p1.prov[0].bbox.t = 580.0

        # Page 2: Docling emits the footnote (physically at bottom) BEFORE the body
        # text (physically near the top) — the order that triggers the regression.
        footnote_p2 = make_sized_text_item("3See my earlier work on this topic.",
                                            charspan_length=35, bbox_height=8.0,
                                            page_no=2)
        footnote_p2.prov[0].bbox.t = 80.0   # bottom of 700-unit page

        body_p2 = make_sized_text_item("The central thesis now stands complete.",
                                        charspan_length=39, bbox_height=10.0,
                                        page_no=2)
        body_p2.prov[0].bbox.t = 580.0      # top of page

        # Items in Docling's wrong emission order: footnote before body text on p.2
        texts = [body_p1, footnote_p2, body_p2]
        parser = make_parser(texts, include_notes=False, min_footnote_chars=100,
                             page_height=700.0)
        docs, _ = parser.run()

        assert any("central thesis" in d for d in docs), (
            "Body text on p.2 was swept as a footnote because Docling emitted "
            "the footnote before the body text and H3 fired unconditionally."
        )

    def test_two_column_page_preserves_emission_order(self) -> None:
        """A page whose items span two columns (bbox.l spread > 100 pts) must not be
        Y-sorted — Docling's emission order is preserved to avoid interleaving columns.

        Setup: Docling emits the left-column item first, then the right-column item.
        The right item has a higher bbox.t (480 > 400), so a naive Y-sort would move
        it before the left item.  With multi-column detection the sort is skipped and
        the original emission order (left before right) must be preserved.
        """
        left_item = make_sized_text_item("Left column text on this page.",
                                         charspan_length=30, bbox_height=10.0, page_no=1)
        left_item.prov[0].bbox.t = 400.0
        left_item.prov[0].bbox.l = 50.0    # left column

        right_item = make_sized_text_item("Right column text on this page.",
                                          charspan_length=31, bbox_height=10.0, page_no=1)
        right_item.prov[0].bbox.t = 480.0  # higher t → naive Y-sort puts this first
        right_item.prov[0].bbox.l = 350.0  # right column; spread = 350-50 = 300 > 100

        texts = [left_item, right_item]
        parser = make_parser(texts, include_notes=False)
        docs, _ = parser.run()

        left_idx = next(i for i, d in enumerate(docs) if "Left column" in d)
        right_idx = next(i for i, d in enumerate(docs) if "Right column" in d)
        assert left_idx < right_idx, (
            "Two-column page was Y-sorted, interleaving left and right columns. "
            "Items with bbox.l spread > 100 should preserve Docling's emission order."
        )

    def test_centered_section_header_does_not_block_sort(self) -> None:
        """A centered section header with a large bbox.l must not trigger multi-column
        detection.  Only TEXT items should contribute to the l-spread calculation.

        Setup mirrors the real p.293 problem.  A body text item on page 1 sets
        prev_text_candidate=True (ends mid-sentence).  On page 2, Docling emits a
        footnote (physically at bottom, low bbox.t) before the body text (physically
        at the top, high bbox.t).  A SECTION_HEADER with bbox.l=178 sits on page 2,
        pushing the ALL-item l-spread above 100.

        Without the TEXT-only filter: sort skipped → footnote processed first →
        H1 fires (prev_text_candidate=True from p.1, digit-start, lower half) →
        found_note_this_page=True → H3 sweeps body text → body text lost.

        With the fix: only TEXT items contribute to the l-spread → spread < 100 →
        sort fires → body text processed first → survives.
        """
        # Page 1: long mid-sentence body text → sets prev_text_candidate=True
        prev_body = make_sized_text_item("A" * 150, charspan_length=150,
                                         bbox_height=10.0, page_no=1)
        prev_body.prov[0].bbox.t = 400.0
        prev_body.prov[0].bbox.l = 61.0

        # Page 2: footnote emitted first by Docling (physically at bottom, low t)
        footnote = make_sized_text_item("6See footnote 4.", charspan_length=16,
                                        bbox_height=10.0, page_no=2)
        footnote.prov[0].bbox.t = 80.0   # lower half of 700-unit page → H1 fires
        footnote.prov[0].bbox.l = 68.0

        # Page 2: body text emitted second (physically at top, high t)
        body = make_sized_text_item(
            "Thus both the problems are solved. Yet there seems to be room.",
            charspan_length=62, bbox_height=10.0, page_no=2)
        body.prov[0].bbox.t = 550.0
        body.prov[0].bbox.l = 61.0

        # Page 2: centered section header — large l, not a second column
        header = make_section_header("CORROBORATION", page_no=2)
        header.prov[0].bbox.t = 580.0
        header.prov[0].bbox.l = 178.0   # all-item spread: 178-61=117 > 100

        # Docling emits: prev_body, footnote, body, header (footnote before body)
        texts = [prev_body, footnote, body, header]
        parser = make_parser(texts, include_notes=False, min_footnote_chars=100,
                             page_height=700.0)
        docs, _ = parser.run()

        assert any("problems are solved" in d for d in docs), (
            "Body text was swept as a footnote. The centered section header "
            "triggered false multi-column detection and blocked the sort."
        )

    def test_section_header_sets_text_seen_enables_h4(self) -> None:
        """A section header followed by a digit+uppercase footnote must trigger H4.
        Section headers now count as text seen on this page."""
        section = make_section_header_at("III. The Arrow of Time",
                                         bbox_t=600.0, bbox_height=10.0)
        footnote = make_sized_text_item("5To make all this quite clear we write",
                                        charspan_length=38, bbox_height=10.0)
        footnote.prov[0].bbox.t = 80.0
        texts = [section, footnote]
        parser = make_parser(texts, min_footnote_chars=100, include_notes=False)
        docs, _ = parser.run()
        assert all("make all this quite clear" not in d for d in docs)

    def test_list_item_sets_text_seen_enables_h4(self) -> None:
        """A list item followed by a digit+uppercase footnote must trigger H4.
        List items now count as text seen on this page."""
        list_item = make_doc_item(TextItem, DocItemLabel.LIST_ITEM.value,
                                  "(a) First condition of the argument")
        list_item.prov[0].bbox.t = 600.0
        footnote = make_sized_text_item("3See my earlier work on this topic.",
                                        charspan_length=35, bbox_height=10.0)
        footnote.prov[0].bbox.t = 80.0
        texts = [list_item, footnote]
        parser = make_parser(texts, min_footnote_chars=100, include_notes=False)
        docs, _ = parser.run()
        assert all("earlier work" not in d for d in docs)

    def test_formula_sets_text_seen_enables_h4(self) -> None:
        """A formula followed by a digit+uppercase footnote must trigger H4.
        Formula items now count as text seen on this page."""
        formula = make_doc_item(TextItem, DocItemLabel.FORMULA.value, "p(x|z) >= p(x)")
        formula.prov[0].bbox.t = 600.0
        footnote = make_sized_text_item("7See the derivation in chapter two.",
                                        charspan_length=35, bbox_height=10.0)
        footnote.prov[0].bbox.t = 80.0
        texts = [formula, footnote]
        parser = make_parser(texts, min_footnote_chars=100, include_notes=False)
        docs, _ = parser.run()
        assert all("derivation in chapter two" not in d for d in docs)

    def test_notes_section_header_triggers_endnote_path(self) -> None:
        """A 'Notes' SECTION_HEADER causes subsequent digit+alpha TEXT items to be
        classified as endnotes, even without small text or body text on the page."""
        texts = [
            make_text_item("Body text on some earlier page.", page_no=1),
            make_section_header("Notes", page_no=2),
            make_text_item("9 Manning asserts that what makes an illegal seizure of power...",
                           page_no=2),
        ]
        parser = make_parser(texts, include_notes=False)
        docs, meta = parser.run()
        assert all("Manning asserts" not in d for d in docs)

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
        """A TEXT item reclassified as footnote must show 'text → footnote:' on its line.
        Page footer provides single_line_height; footnote item is smaller than body text."""
        long_mid = "A" * 100
        footnote_item = make_sized_text_item(
            "1 This is a citation reference.",
            charspan_length=31, bbox_height=8.0,
        )
        texts = [
            make_text_item(long_mid),
            footnote_item,
            make_page_footer("1", page_no=1),   # gives single_line_height=10.0
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
        chunks = parser._get_processed_texts()
        assert chunks[0].meta['page_#'] == 'i'

    def test_page_meta_falls_back_to_physical_number_without_labels(self) -> None:
        """Without page labels, chunk metadata falls back to the physical page number string."""
        texts = [make_text_item("Body text.", page_no=42)]
        parser = make_parser(texts)
        chunks = parser._get_processed_texts()
        assert chunks[0].meta['page_#'] == '42'

    def test_empty_string_label_falls_back_to_physical_number(self) -> None:
        """pypdfium2 returns '' for PDFs with no page label table; must still show physical number."""
        texts = [make_text_item("Body text.", page_no=5)]
        parser = make_parser(texts, page_labels={4: ''})   # empty string, not None
        chunks = parser._get_processed_texts()
        assert chunks[0].meta['page_#'] == '5'

    def test_arabic_label_stored_verbatim_in_meta(self) -> None:
        """Arabic page labels are stored verbatim — the body of a book with 40 front-matter pages."""
        texts = [make_text_item("Body text.", page_no=41)]
        parser = make_parser(texts, page_labels={40: '1'})   # Docling page 41 → index 40 → '1'
        chunks = parser._get_processed_texts()
        assert chunks[0].meta['page_#'] == '1'

    def test_front_matter_pages_skipped_when_enabled(self) -> None:
        """Pages with Roman numeral labels are excluded when skip_front_matter=True."""
        texts = [
            make_text_item("Front matter text.", page_no=1),
            make_text_item("Body text.", page_no=2),
        ]
        parser = make_parser(texts, page_labels={0: 'i', 1: '1'}, skip_front_matter=True)
        docs, meta = parser.run()
        assert len(docs) == 1
        assert meta[0]['page_#'] == '1'

    def test_front_matter_included_when_skip_front_matter_false(self) -> None:
        """Front matter pages are kept when skip_front_matter=False (the default)."""
        texts = [
            make_text_item("Front matter text.", page_no=1),
            make_text_item("Body text.", page_no=2),
        ]
        parser = make_parser(texts, page_labels={0: 'i', 1: '1'}, skip_front_matter=False)
        docs, _ = parser.run()
        assert len(docs) == 2

    def test_both_page_numbers_present_in_meta(self) -> None:
        """Both the PDF label and the physical page number appear in chunk metadata."""
        texts = [make_text_item("Body text.", page_no=41)]
        parser = make_parser(texts, page_labels={40: '1'})
        chunks = parser._get_processed_texts()
        assert chunks[0].meta['page_#'] == '1'
        assert chunks[0].meta['physical_page_#'] == '41'

    def test_physical_page_matches_docling_page_no(self) -> None:
        """physical_page_# always reflects Docling's page_no regardless of label."""
        texts = [make_text_item("Front matter.", page_no=5)]
        parser = make_parser(texts, page_labels={4: 'v'})
        chunks = parser._get_processed_texts()
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
        docs, _ = parser.run()
        assert len(docs) == 1
        assert "Body text." in docs[0]


# --- TestFindIndexStartPage ---

class TestFindIndexStartPage:
    """Tests for DoclingParser._find_index_start_page().

    The method scans self._doc.texts for two signals that indicate the start of
    a back-matter index section:

      - PAGE_HEADER: text contains 'index' after all whitespace is stripped
        (handles OCR-spaced titles like 'I N DEX OF SUBJ ECTS').
      - SECTION_HEADER: text contains 'index', 'indexes', 'indices', or 'indice'
        as a complete word (word-boundary anchored, case-insensitive).

    Both signals are subject to a position gate: signals in the first 70% of
    the document are ignored to prevent false positives (e.g. a chapter titled
    'Indexical Reference' early in the book). Total page count is estimated
    from the highest page_no seen across all items in self._doc.texts.
    """

    def test_returns_none_when_no_index(self) -> None:
        """No index signals present → None."""
        texts = [
            make_text_item("Body text.", page_no=1),
            make_text_item("More body.", page_no=2),
        ]
        parser = make_parser(texts)
        assert parser._find_index_start_page() is None

    @pytest.mark.parametrize("header_text, page_no, expected", [
        ("Index",                 90, 90),  # plain 'Index'
        ("I N DEX OF SUBJ ECTS", 90, 90),  # OCR-spaced — strip whitespace to detect
        ("Index of Names",        95, 95),  # multi-word variant
    ])
    def test_page_header_detected(self, header_text: str, page_no: int, expected: int) -> None:
        """PAGE_HEADER containing 'index' (after whitespace stripping) in the last 30% is detected."""
        texts = [
            make_text_item("Body.", page_no=1),
            make_text_item("More body.", page_no=100),
            make_page_header(header_text, page_no=page_no),
        ]
        parser = make_parser(texts)
        assert parser._find_index_start_page() == expected

    @pytest.mark.parametrize("header_text, page_no, expected", [
        ("Index",          90, 90),  # exact word match
        ("Indices",        90, 90),  # plural variant
        ("Index of Names", 92, 92),  # multi-word, 'index' as whole word
    ])
    def test_section_header_detected(self, header_text: str, page_no: int, expected: int) -> None:
        """SECTION_HEADER with word-boundary 'index' match in the last 30% is detected."""
        texts = [
            make_text_item("Body.", page_no=1),
            make_text_item("More body.", page_no=100),
            make_section_header(header_text, page_no=page_no),
        ]
        parser = make_parser(texts)
        assert parser._find_index_start_page() == expected

    def test_multiple_signals_returns_minimum_page(self) -> None:
        """When both SECTION_HEADER and PAGE_HEADER fire, the earliest page wins.

        Typical structure: 'Indices' section header appears on the first index page
        (page 88); the running page header 'I N DEX OF SUBJ ECTS' appears from
        page 89 onwards. The section header fires first, so 88 is returned.
        """
        texts = [
            make_text_item("Body.", page_no=1),
            make_text_item("More body.", page_no=100),
            make_section_header("Indices", page_no=88),
            make_page_header("I N DEX OF SUBJ ECTS", page_no=89),
        ]
        parser = make_parser(texts)
        assert parser._find_index_start_page() == 88

    def test_position_gate_rejects_early_section_header(self) -> None:
        """A section header 'Indices' in the first 70% of the book is ignored.

        Prevents false positives from chapter titles that happen to contain
        the word 'index' early in the text. Here page 5 of 100 (5%) is well
        within the first 70%, so the gate fires and None is returned.
        """
        texts = [
            make_text_item("Body.", page_no=1),
            make_text_item("More body.", page_no=100),  # max_page=100; gate threshold=70
            make_section_header("Indices", page_no=5),  # page 5 < 70 → rejected
        ]
        parser = make_parser(texts)
        assert parser._find_index_start_page() is None

    def test_position_gate_rejects_early_page_header(self) -> None:
        """A PAGE_HEADER containing 'index' in the first 70% of the book is ignored."""
        texts = [
            make_text_item("Body.", page_no=1),
            make_text_item("More body.", page_no=100),
            make_page_header("Index", page_no=10),  # page 10 < 70 → rejected
        ]
        parser = make_parser(texts)
        assert parser._find_index_start_page() is None

    def test_indexical_section_header_not_matched(self) -> None:
        """'Indexical Reference' contains 'index' as a prefix but not as a whole word.

        The word-boundary regex \\bindex\\b must not match inside 'Indexical'.
        """
        texts = [
            make_text_item("Body.", page_no=1),
            make_text_item("More body.", page_no=100),
            make_section_header("Indexical Reference", page_no=90),
        ]
        parser = make_parser(texts)
        assert parser._find_index_start_page() is None

    def test_page_footer_index_not_matched(self) -> None:
        """PAGE_FOOTER items are not inspected — only PAGE_HEADER and SECTION_HEADER."""
        texts = [
            make_text_item("Body.", page_no=1),
            make_text_item("More body.", page_no=100),
            make_page_footer("Index", page_no=90),
        ]
        parser = make_parser(texts)
        assert parser._find_index_start_page() is None

    def test_body_text_item_index_not_matched(self) -> None:
        """A regular TEXT item containing the word 'index' must not trigger detection."""
        texts = [
            make_text_item("Body.", page_no=1),
            make_text_item("See the index for more.", page_no=100),
        ]
        parser = make_parser(texts)
        assert parser._find_index_start_page() is None


# --- TestSkipIndex ---

class TestSkipIndex:
    """Integration tests for skip_index=True end-to-end through run()."""

    def test_skip_index_removes_index_and_later_pages(self) -> None:
        """With skip_index=True, the detected index page and all pages after are excluded."""
        texts = [
            make_text_item("Body text on page one.", page_no=1),
            make_text_item("More body on page two.", page_no=2),
            make_text_item("More body.", page_no=100),   # establishes max_page=100
            make_page_header("Index", page_no=90),        # signals index start at page 90
            make_text_item("Subject: Popper, 45, 78.", page_no=90),
            make_text_item("Still index content.", page_no=95),
        ]
        parser = make_parser(texts, skip_index=True)
        docs, _ = parser.run()
        assert any("Body text on page one." in d for d in docs)
        assert all("Subject: Popper" not in d for d in docs)
        assert all("Still index content." not in d for d in docs)

    def test_skip_index_false_keeps_index_pages(self) -> None:
        """With skip_index=False (explicit), index pages are included in output."""
        texts = [
            make_text_item("Body text.", page_no=1),
            make_text_item("More body.", page_no=100),
            make_page_header("Index", page_no=90),
            make_text_item("Subject: Popper, 45, 78.", page_no=90),
        ]
        parser = make_parser(texts, skip_index=False)
        docs, _ = parser.run()
        assert any("Subject: Popper" in d for d in docs)

    def test_skip_index_default_is_false(self) -> None:
        """Without the skip_index argument, default is False and index pages are kept."""
        texts = [
            make_text_item("Body text.", page_no=1),
            make_text_item("More body.", page_no=100),
            make_page_header("Index", page_no=90),
            make_text_item("Subject: Popper, 45, 78.", page_no=90),
        ]
        parser = make_parser(texts)   # no skip_index argument
        docs, _ = parser.run()
        assert any("Subject: Popper" in d for d in docs)

    def test_skip_index_no_index_found_keeps_all_content(self) -> None:
        """With skip_index=True but no detectable index, nothing is dropped."""
        texts = [
            make_text_item("Body text on page one.", page_no=1),
            make_text_item("Body text on page two.", page_no=2),
        ]
        parser = make_parser(texts, skip_index=True)
        docs, _ = parser.run()
        assert any("Body text on page one." in d for d in docs)
        assert any("Body text on page two." in d for d in docs)
