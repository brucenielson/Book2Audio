from pathlib import Path
from docling_core.types.doc.document import (SectionHeaderItem,
                                             ListItem,
                                             TextItem,
                                             DocItem,
                                             DocItemLabel,
                                             DoclingDocument)
from docling.document_converter import DocumentConverter
from typing import List
import re
from utils.general_utils import (is_sentence_end,
                           normalize_quotes,
                           normalize_whitespace,
                           fix_apostrophes,
                           normalize_ligatures,
                           fix_encoding_artifacts,
                           fix_bracket_spacing,
                           fix_punctuation_spacing,
                           strip_footnote_numbers)


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


def is_section_header(text: DocItem | None) -> bool:
    """Check if a DocItem is a section header.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a section header, False otherwise.
    """
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == DocItemLabel.SECTION_HEADER.value


def is_page_footer(text: DocItem | None) -> bool:
    """Check if a DocItem is a page footer.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a page footer, False otherwise.
    """
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == DocItemLabel.PAGE_FOOTER.value


def is_page_header(text: DocItem | None) -> bool:
    """Check if a DocItem is a page header.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a page header, False otherwise.
    """
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == DocItemLabel.PAGE_HEADER.value


def is_footnote(text: DocItem | None) -> bool:
    """Check if a DocItem is a footnote.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a footnote, False otherwise.
    """
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == DocItemLabel.FOOTNOTE.value


def is_list_item(text: DocItem | None) -> bool:
    """Check if a DocItem is a list item.

    Args:
        text: The DocItem to check, or None.

    Returns:
        True if the item is a list item, False otherwise.
    """
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
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
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
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
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
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
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
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
    if not isinstance(doc_item, TextItem):
        return False
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
    if not isinstance(item, (SectionHeaderItem, ListItem, TextItem)):
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


def combine_paragraphs(p1_str: str, p2_str: str) -> str:
    """Combine two paragraph strings into one.

    If the first paragraph ends with sentence-ending punctuation, the two are
    joined with a newline. Otherwise, they are joined with a space, treating
    them as a continuation of the same sentence.

    Args:
        p1_str: The first paragraph string.
        p2_str: The second paragraph string.

    Returns:
        The combined paragraph string, stripped of leading and trailing whitespace.
    """
    p1_str = p1_str.strip()
    p2_str = p2_str.strip()
    if is_sentence_end(p1_str):
        combined = p1_str + "\n" + p2_str
    else:
        combined = p1_str + " " + p2_str
    return combined.strip()


def is_roman_numeral(s: str) -> bool:
    """Check if a string is a Roman numeral.

    The check is case-insensitive and matches standard Roman numerals
    from I to MMMCMXCIX.

    Args:
        s: The string to check.

    Returns:
        True if the string is a valid Roman numeral, False otherwise.
    """
    roman_numeral_pattern: str = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))


def get_current_page(text: DocItem,
                     combined_paragraph: str,
                     current_page: int | None) -> int | None:
    """Determine the current page number based on the given DocItem.

    Returns the item's page number only if no page has been recorded yet
    or if no paragraph is currently being accumulated. Otherwise returns
    the existing page number unchanged.

    Args:
        text: The DocItem whose provenance page number may be used.
        combined_paragraph: The paragraph string currently being accumulated.
        current_page: The current page number, or None if not yet set.

    Returns:
        The updated page number, or the existing one if unchanged.
    """
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
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
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return True
    return any([
        is_page_footer(text),
        is_page_header(text),
        is_roman_numeral(text.text)
    ])


def clean_text_pdf(p_str: str) -> str:
    """Clean and normalise a paragraph string extracted from a PDF.

    Applies a pipeline of normalisation steps in order:
    whitespace, hyphens, ligatures, quotes, punctuation spacing,
    bracket spacing, apostrophes, and footnote number stripping.

    Args:
        p_str: The raw paragraph string to clean.

    Returns:
        The cleaned and normalised string.
    """
    p_str = normalize_whitespace(p_str)
    # p_str = _normalize_hyphens(p_str)
    p_str = normalize_ligatures(p_str)
    p_str = fix_encoding_artifacts(p_str)
    p_str = normalize_quotes(p_str)
    p_str = fix_punctuation_spacing(p_str)
    p_str = fix_bracket_spacing(p_str)
    p_str = fix_apostrophes(p_str)
    p_str = strip_footnote_numbers(p_str)
    return p_str.strip()
