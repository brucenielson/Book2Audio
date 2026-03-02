from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem, DocItemLabel
from typing import List
import re


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


def is_ends_with_punctuation(text: str) -> bool:
    """Check if a string ends with sentence-ending punctuation.

    Args:
        text: The string to check.

    Returns:
        True if the string ends with a period, question mark, or exclamation point.
    """
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


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


def is_sentence_end(text: str) -> bool:
    """Check if a string ends with a complete sentence.

    Handles standard punctuation as well as closing brackets and quotes
    that follow sentence-ending punctuation.

    Args:
        text: The string to check.

    Returns:
        True if the string appears to end a complete sentence.
    """
    has_end_punctuation: bool = is_ends_with_punctuation(text)
    # Does it end with a closing bracket, quote, etc.?
    ends_with_bracket: bool = (text.endswith(")")
                               or text.endswith("]")
                               or text.endswith("}")
                               or text.endswith("\"")
                               or text.endswith("\'"))
    return (has_end_punctuation or
            (ends_with_bracket and is_ends_with_punctuation(text[0:-1])))


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


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple consecutive whitespace characters into a single space.

    Also strips leading and trailing whitespace.

    Args:
        text: The string to process.

    Returns:
        The string with all whitespace runs collapsed to a single space.
    """
    # Remove extra whitespace in the middle of the text
    return ' '.join(text.split())


def combine_paragraphs(p1_str: str, p2_str: str) -> str:
    """Combine two paragraph strings into one.

    If the first paragraph ends with sentence-ending punctuation, the two are
    joined with a newline. Otherwise they are joined with a space, treating
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


def _normalize_ligatures(p_str: str) -> str:
    """Replace common OCR ligature characters with their letter equivalents.

    Args:
        p_str: The string to process.

    Returns:
        The string with ligatures replaced.
    """
    p_str = p_str.replace('ﬁ', 'fi')
    p_str = p_str.replace('ﬂ', 'fl')
    p_str = p_str.replace('ﬀ', 'ff')
    p_str = p_str.replace('ﬃ', 'ffi')
    p_str = p_str.replace('ﬄ', 'ffl')
    p_str = p_str.replace('ﬅ', 'st')
    return p_str


def _normalize_quotes(p_str: str) -> str:
    """Replace curly/smart quotes with straight ASCII equivalents.

    Args:
        p_str: The string to process.

    Returns:
        The string with smart quotes replaced by straight quotes.
    """
    p_str = p_str.replace('\u201c', '"').replace('\u201d', '"')  # " "
    p_str = p_str.replace('\u2018', "'").replace('\u2019', "'")  # ' '
    return p_str


def _normalize_whitespace(p_str: str) -> str:
    """Strip and collapse whitespace in a string.

    Converts the input to a string, strips leading and trailing whitespace,
    ensures UTF-8 encoding, and collapses internal whitespace runs to single spaces.

    Args:
        p_str: The string to process.

    Returns:
        The normalised string.
    """
    p_str = str(p_str).strip()  # Convert text to a string and remove leading/trailing whitespace
    p_str = p_str.encode('utf-8').decode('utf-8')
    return remove_extra_whitespace(p_str)


def _fix_punctuation_spacing(p_str: str) -> str:
    """Remove erroneous whitespace around punctuation marks.

    Handles spaces before periods, commas, question marks, exclamation points,
    and spaces between sentence-ending punctuation and closing quotes.

    Args:
        p_str: The string to process.

    Returns:
        The string with punctuation spacing corrected.
    """
    p_str = re.sub(r"([.!?]) '", r"\1'", p_str)  # Remove the space between punctuation (.!?) and '
    p_str = re.sub(r'([.!?]) "', r'\1"', p_str)  # Remove the space between punctuation (.!?) and "
    p_str = re.sub(r'\s+,', ',', p_str)  # Remove whitespace before a comma
    p_str = re.sub(r'(?<=\s)\.([a-zA-Z])', r'\1', p_str)  # Remove a period that follows a whitespace and comes before a letter
    p_str = re.sub(r'\s+\.', '.', p_str)  # Remove any whitespace before a period
    p_str = re.sub(r'\s+\?', '?', p_str)  # Remove any whitespace before a question mark
    p_str = re.sub(r'\s+!', '!', p_str)  # Remove any whitespace before an exclamation point
    return p_str


def _fix_bracket_spacing(p_str: str) -> str:
    """Remove erroneous whitespace inside brackets and parentheses.

    Args:
        p_str: The string to process.

    Returns:
        The string with bracket spacing corrected.
    """
    p_str = re.sub(r'\s+\)', ')', p_str)  # Remove whitespace before a closing parenthesis
    p_str = re.sub(r'\s+]', ']', p_str)  # Remove whitespace before a closing square bracket
    p_str = re.sub(r'\s+}', '}', p_str)  # Remove whitespace before a closing curly brace
    p_str = re.sub(r'\(\s+', '(', p_str)  # Remove whitespace after an opening parenthesis
    p_str = re.sub(r'\[\s+', '[', p_str)  # Remove whitespace after an opening square bracket
    p_str = re.sub(r'\{\s+', '{', p_str)  # Remove whitespace after an opening curly brace
    return p_str


def _fix_apostrophes(p_str: str) -> str:
    """Fix erroneous whitespace around possessive apostrophes.

    Handles two OCR artifacts: a space between an apostrophe and the following
    's', and a space before a possessive 's.

    Args:
        p_str: The string to process.

    Returns:
        The string with apostrophe spacing corrected.
    """
    # Remove white space between an ' and an s if there is a white space after the s (i.e. possessive apostrophe) or if this is a punctuation mark {., !, ?, :}
    p_str = re.sub(r"'\s+s(\s|[.,!?;:])", r"'s\1", p_str)
    p_str = re.sub(r"\s+'s(\s|$)", r"'s\1", p_str)  # Remove space before 's (possessive)
    return p_str


def _strip_footnote_numbers(p_str: str) -> str:
    """Remove trailing footnote reference numbers from a string.

    Strips digits from the end of the string until the string ends with
    valid sentence-ending punctuation or no more digits remain.

    Args:
        p_str: The string to process.

    Returns:
        The string with trailing footnote numbers removed.
    """
    # Remove footnote numbers at end of a sentence. Check for a digit at the end and drop it
    # until there are no more digits or the sentence is now a valid end of a sentence.
    while p_str and not is_sentence_end(p_str):
        last_char: str = p_str[-1]
        if not last_char.isdigit():
            break
        p_str = p_str[:-1].strip()
    return p_str


def _normalize_hyphens(p_str: str) -> str:
    """Remove soft hyphens (SHY, U+00AD) from a string.

    Soft hyphens are invisible line-break hints inserted by typesetters.
    In OCR'd text they appear where a word was broken across lines and
    should be removed to restore the original word.

    Args:
        p_str: The string to process.

    Returns:
        The string with soft hyphens removed.
    """
    p_str = p_str.replace("\u00ad", "")  # Remove soft hyphen (SHY) - line-break hint, not a real hyphen
    return p_str


def clean_text(p_str: str) -> str:
    """Clean and normalise a paragraph string extracted from a PDF.

    Applies a pipeline of normalisation steps in order:
    whitespace, hyphens, ligatures, quotes, punctuation spacing,
    bracket spacing, apostrophes, and footnote number stripping.

    Args:
        p_str: The raw paragraph string to clean.

    Returns:
        The cleaned and normalised string.
    """
    p_str = _normalize_whitespace(p_str)
    p_str = _normalize_hyphens(p_str)
    p_str = _normalize_ligatures(p_str)
    p_str = _normalize_quotes(p_str)
    p_str = _fix_punctuation_spacing(p_str)
    p_str = _fix_bracket_spacing(p_str)
    p_str = _fix_apostrophes(p_str)
    p_str = _strip_footnote_numbers(p_str)
    return p_str.strip()
