"""Utility functions for working with Docling document objects."""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard

from docling_core.types.doc.document import (TextItem,
                                             DocItem,
                                             DocItemLabel,
                                             DoclingDocument)
from docling.document_converter import DocumentConverter


def load_as_document(file_path: str | Path) -> DoclingDocument:
    """Load a document file and return it as a DoclingDocument.

    If a cached JSON file exists at the same path (with a .json extension),
    it will be loaded directly instead of re-converting the source file.
    Otherwise, the file is converted using DocumentConverter and the result
    is saved as JSON for future use.

    Args:
        file_path: Path to the source document file, as a string or Path object
                   (e.g. a PDF).

    Returns:
        A DoclingDocument representing the parsed document.
    """
    json_path: Path = Path(file_path).with_suffix('.json')
    if json_path.exists():
        return DoclingDocument.load_from_json(json_path)
    converter: DocumentConverter = DocumentConverter()
    result = converter.convert(file_path)
    book: DoclingDocument = result.document
    book.save_as_json(json_path)
    return book


def is_text_bearing(item: DocItem | None) -> TypeGuard[TextItem]:
    """Check if a DocItem is a text-bearing subclass (i.e. TextItem).

    Note that SectionHeaderItem and ListItem are inherited from TextItem.
    Acts as a TypeGuard — callers that branch on this function have their
    item narrowed to TextItem in the True branch.

    Args:
        item: The DocItem to check, or None.

    Returns:
        True if item is an instance of TextItem, False otherwise.
    """
    return isinstance(item, TextItem)


def is_section_header(text: DocItem | None) -> bool:
    """Check if a DocItem is a section header.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a section header, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return text.label == DocItemLabel.SECTION_HEADER


def is_page_footer(text: DocItem | None) -> bool:
    """Check if a DocItem is a page footer.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a page footer, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return text.label == DocItemLabel.PAGE_FOOTER


def is_page_header(text: DocItem | None) -> bool:
    """Check if a DocItem is a page header.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a page header, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return text.label == DocItemLabel.PAGE_HEADER


def is_footnote(text: DocItem | None) -> bool:
    """Check if a DocItem is a footnote.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a footnote, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return text.label == DocItemLabel.FOOTNOTE


def is_list_item(text: DocItem | None) -> bool:
    """Check if a DocItem is a list item.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a list item, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return text.label == DocItemLabel.LIST_ITEM


# TODO: Check if is_text_break is still needed or can be removed
def is_text_break(text: DocItem | None) -> bool:
    """Check if a DocItem represents a break in the main text flow.

    A text break is a page header, section header, or footnote — any element
    that interrupts the flow of body text.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a text break, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return is_page_header(text) or is_section_header(text) or is_footnote(text)


def is_page_not_text(text: DocItem | None) -> bool:
    """Check if a DocItem is not a body text element.

    Returns True for items that are not regular text, list items, or formulas.
    Also returns True for None or non-text DocItem subclasses.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is not a body text element, False otherwise.
    """
    if not is_text_bearing(text):
        return True
    return text.label not in [DocItemLabel.TEXT, DocItemLabel.LIST_ITEM, DocItemLabel.FORMULA]


def is_page_text(text: DocItem | None) -> bool:
    """Check if a DocItem is a body text element.

    Returns True for items that are regular text, list items, or formulas.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a body text element, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return not is_page_not_text(text)


def is_too_short(doc_item: DocItem, threshold: int = 2) -> bool:
    """Check if a TextItem's text is too short to be meaningful.

    Only applies to TextItem instances. Non-TextItem DocItems always return False.

    Args:
        doc_item: The DocItem to check.
        threshold: Maximum character count to consider too short. Defaults to 2.

    Returns:
        True if the item is a TextItem whose text length is at or below the threshold.
    """
    if not is_text_bearing(doc_item):
        return False
    assert isinstance(doc_item, TextItem)
    return len(doc_item.text) <= threshold


def is_text_item(item: DocItem | None) -> bool:
    """Check if a DocItem is a body text item suitable for paragraph processing.

    Returns False for section headers, page footers, page headers, None,
    and any non-text DocItem subclasses.

    Args:
        item: The DocItem to check, or None.

    Returns:
        True if the item is a processable body text item.
    """
    if not is_text_bearing(item):
        return False
    return not (is_section_header(item)
                or is_page_footer(item)
                or is_page_header(item))


def get_next_text(texts: list[DocItem], i: int) -> DocItem | None:
    """Find the next body text item in a list of DocItems after index i.

    Args:
        texts: The list of DocItems to search.
        i: The current index. The search starts from i + 1.

    Returns:
        The next DocItem that passes is_text_item, or None if not found.
    """
    # Seek through the list of texts to find the next text item using is_text_item
    # Should return None if no more text items are found
    for j in range(i + 1, len(texts)):
        if is_text_item(texts[j]):
            return texts[j]
    return None


def get_current_page(text: DocItem,
                     combined_paragraph: str,
                     current_page: int | None) -> int | None:
    """Determine the current page number based on the given DocItem.

    Returns the item's page number only if no page has been recorded yet
    or if no paragraph is currently being accumulated. Otherwise, returns
    the existing page number unchanged.

    Args:
        text: The DocItem whose provenance page number may be used.
        combined_paragraph: The paragraph string currently being accumulated.
        current_page: The current page number, or None if not yet set.

    Returns:
        The updated page number, or the existing one if unchanged.
    """
    if not is_text_bearing(text):
        return current_page
    # noinspection PyTypeHints
    return text.prov[0].page_no if current_page is None or combined_paragraph == "" else current_page


def compute_single_line_height(doc: DoclingDocument) -> float:
    """Compute the median bbox.height of page header and footer items.

    Page headers and footers are reliably single-line items, making their
    bbox.height a good baseline for the height of one line of text.

    Args:
        doc: The DoclingDocument to analyse.

    Returns:
        The median bbox.height of page headers and footers, or 0.0 if none found.
    """
    heights: list[float] = []
    for item in doc.texts:
        if not is_text_bearing(item):
            continue
        if not (is_page_header(item) or is_page_footer(item)):
            continue
        if not item.prov:
            continue
        prov = item.prov[0]
        if prov.bbox is None:
            continue
        heights.append(prov.bbox.height)
    if not heights:
        return 0.0
    heights.sort()
    return heights[len(heights) // 2]


def compute_median_chars_per_line(items: list[TextItem], single_line_height: float,
                                   min_charspan: int = 100) -> float:
    """Compute the median characters-per-estimated-line across a list of TextItems.

    For each item, estimates the number of lines as bbox.height / single_line_height,
    then divides charspan_length by that estimate. The median of this value across
    all body text items represents normal characters per line for the document.

    Only items with a charspan of at least min_charspan are included, so that short
    items (headings, list items, page numbers etc.) do not drag the median down and
    make the threshold comparison unreliable.

    Args:
        items: The TextItems to analyse.
        single_line_height: The height of a single line, from compute_single_line_height().
        min_charspan: Minimum charspan length to include in the median calculation.
                      Defaults to 100.

    Returns:
        The median chars-per-estimated-line, or 0.0 if no valid items are found.
    """
    if single_line_height <= 0:
        return 0.0
    ratios: list[float] = []
    for item in items:
        if not item.prov:
            continue
        prov = item.prov[0]
        if prov.bbox is None:
            continue
        if prov.bbox.height <= 0:
            continue
        charspan_length: int = prov.charspan[1] - prov.charspan[0]
        if charspan_length < min_charspan:
            continue
        estimated_lines: float = prov.bbox.height / single_line_height
        ratios.append(charspan_length / estimated_lines)
    if not ratios:
        return 0.0
    ratios.sort()
    return ratios[len(ratios) // 2]


def is_small_text(item: TextItem, single_line_height: float,
                  median_chars_per_line: float, threshold: float = 1.25) -> bool:
    """Return True if a TextItem's font is significantly smaller than the document norm.

    Uses characters-per-estimated-line as a proxy for font size. Smaller fonts
    pack more characters into each estimated line, so items with significantly
    more chars per estimated line than the document median are likely in smaller
    text. This approach works for both short and long footnotes.

    Args:
        item: The TextItem to check.
        single_line_height: The height of a single line, from compute_single_line_height().
        median_chars_per_line: The median chars-per-estimated-line for the document,
                               from compute_median_chars_per_line().
        threshold: Items above this multiple of the median are considered small
                   text. Defaults to 1.25.

    Returns:
        True if the item's chars-per-estimated-line exceeds median_chars_per_line * threshold.
    """
    if not item.prov or single_line_height <= 0 or median_chars_per_line <= 0:
        return False
    prov = item.prov[0]
    if prov.bbox is None:
        return False
    if prov.bbox.height <= 0:
        return False
    charspan_length: int = prov.charspan[1] - prov.charspan[0]
    if charspan_length <= 0:
        return False
    estimated_lines: float = prov.bbox.height / single_line_height
    chars_per_line: float = charspan_length / estimated_lines
    return chars_per_line > median_chars_per_line * threshold


def is_single_line(item: TextItem, single_line_height: float, tolerance: float = 1.3) -> bool:
    """Return True if a TextItem's bbox height is consistent with a single line of text.

    Used to distinguish mislabeled running page headers (always one line) from
    genuine multi-line section headers. If single_line_height is zero or bbox
    data is unavailable, returns False so the caller does not incorrectly skip
    the item.

    Args:
        item: The TextItem to check.
        single_line_height: The median height of one line, from compute_single_line_height().
        tolerance: The item's bbox height must be <= single_line_height * tolerance
                   to be considered single-line. Defaults to 1.5.

    Returns:
        True if the item appears to be a single line of text, False otherwise.
    """
    if single_line_height <= 0 or not item.prov:
        return False
    bbox = item.prov[0].bbox
    return bbox is not None and bbox.height <= single_line_height * tolerance


def should_skip_element(text: DocItem) -> bool:
    """Check if a DocItem should be skipped during paragraph processing.

    Skips page footers, page headers, Roman numerals, and any DocItem
    that is not a recognized text subclass.

    Args:
        text: The DocItem to check.

    Returns:
        True if the element should be skipped, False otherwise.
    """
    if not is_text_bearing(text):
        return True
    assert isinstance(text, TextItem)
    return any([
        is_page_footer(text),
        is_page_header(text)
    ])
