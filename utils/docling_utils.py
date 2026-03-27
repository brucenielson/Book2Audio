from pathlib import Path
from docling_core.types.doc.document import (TextItem,
                                             DocItem,
                                             DocItemLabel,
                                             DoclingDocument)
from docling.document_converter import DocumentConverter
from typing import List

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


def is_text_bearing(item: DocItem | None) -> bool:
    """Check if a DocItem is a text-bearing subclass (i.e. TextItem).
    Note that SectionHeaderItem and ListItem are inherited from TextItem
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
    return text.label == DocItemLabel.SECTION_HEADER.value


def is_page_footer(text: DocItem | None) -> bool:
    """Check if a DocItem is a page footer.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a page footer, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return text.label == DocItemLabel.PAGE_FOOTER.value


def is_page_header(text: DocItem | None) -> bool:
    """Check if a DocItem is a page header.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a page header, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return text.label == DocItemLabel.PAGE_HEADER.value


def is_footnote(text: DocItem | None) -> bool:
    """Check if a DocItem is a footnote.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a footnote, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return text.label == DocItemLabel.FOOTNOTE.value


def is_list_item(text: DocItem | None) -> bool:
    """Check if a DocItem is a list item.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a list item, False otherwise.
    """
    if not is_text_bearing(text):
        return False
    return text.label == DocItemLabel.LIST_ITEM.value


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
    return text.label not in [DocItemLabel.TEXT.value, DocItemLabel.LIST_ITEM.value, DocItemLabel.FORMULA.value]


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


def get_next_text(texts: List[DocItem], i: int) -> DocItem | None:
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


def should_skip_element(text: DocItem) -> bool:
    """Check if a DocItem should be skipped during paragraph processing.

    Skips page footers, page headers, Roman numerals, and any DocItem
    that is not a recognised text subclass.

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
