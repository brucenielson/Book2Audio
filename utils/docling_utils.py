from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem, DocItemLabel
from typing import List
import re



def is_section_header(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == DocItemLabel.SECTION_HEADER.value


def is_page_footer(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == DocItemLabel.PAGE_FOOTER.value


def is_page_header(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == DocItemLabel.PAGE_HEADER.value


def is_footnote(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == DocItemLabel.FOOTNOTE.value


def is_list_item(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == DocItemLabel.LIST_ITEM.value


def is_text_break(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return is_page_header(text) or is_section_header(text) or is_footnote(text)


def is_page_not_text(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return True
    return text.label not in [DocItemLabel.TEXT.value, DocItemLabel.LIST_ITEM.value, DocItemLabel.FORMULA.value]


def is_page_text(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return not is_page_not_text(text)


def is_ends_with_punctuation(text: str) -> bool:
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


def is_too_short(doc_item: DocItem, threshold: int = 2) -> bool:
    if not isinstance(doc_item, TextItem):
        return False
    return len(doc_item.text) <= threshold


def is_sentence_end(text: str) -> bool:
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
    if not isinstance(item, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return not (is_section_header(item)
                or is_page_footer(item)
                or is_page_header(item))


def get_next_text(texts: List[DocItem], i: int) -> DocItem | None:
    # Seek through the list of texts to find the next text item using is_text_item
    # Should return None if no more text items are found
    for j in range(i + 1, len(texts)):
        if j < len(texts) and is_text_item(texts[j]):
            return texts[j]
    return None


def remove_extra_whitespace(text: str) -> str:
    # Remove extra whitespace in the middle of the text
    return ' '.join(text.split())


def combine_paragraphs(p1_str: str, p2_str: str) -> str:
    # If the paragraph ends without final punctuation, combine it with the next paragraph
    combined: str
    if is_sentence_end(p1_str):
        combined = p1_str + "\n" + p2_str
    else:
        combined = p1_str + " " + p2_str
    return combined.strip()


def is_roman_numeral(s: str) -> bool:
    roman_numeral_pattern: str = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))


def get_current_page(text: DocItem,
                     combined_paragraph: str,
                     current_page: int | None) -> int | None:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return current_page
    # noinspection PyTypeHints
    return text.prov[0].page_no if current_page is None or combined_paragraph == "" else current_page


def should_skip_element(text: DocItem) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return True
    return any([
        is_page_footer(text),
        is_page_header(text),
        is_roman_numeral(text.text)
    ])


def clean_text(p_str: str) -> str:
    p_str = str(p_str).strip()  # Convert text to a string and remove leading/trailing whitespace
    p_str = p_str.encode('utf-8').decode('utf-8')
    p_str = re.sub(r'\s+', ' ', p_str).strip()  # Replace multiple whitespace with single space
    p_str = re.sub(r"([.!?]) '", r"\1'", p_str)  # Remove the space between punctuation (.!?) and '
    p_str = re.sub(r'([.!?]) "', r'\1"', p_str)  # Remove the space between punctuation (.!?) and "
    p_str = re.sub(r'\s+\)', ')', p_str)  # Remove whitespace before a closing parenthesis
    p_str = re.sub(r'\s+]', ']', p_str)  # Remove whitespace before a closing square bracket
    p_str = re.sub(r'\s+}', '}', p_str)  # Remove whitespace before a closing curly brace
    p_str = re.sub(r'\s+,', ',', p_str)  # Remove whitespace before a comma
    p_str = re.sub(r'\(\s+', '(', p_str)  # Remove whitespace after an opening parenthesis
    p_str = re.sub(r'\[\s+', '[', p_str)  # Remove whitespace after an opening square bracket
    p_str = re.sub(r'\{\s+', '{', p_str)  # Remove whitespace after an opening curly brace
    p_str = re.sub(r'(?<=\s)\.([a-zA-Z])', r'\1',
                   p_str)  # Remove a period that follows a whitespace and comes before a letter
    p_str = re.sub(r'\s+\.', '.', p_str)  # Remove any whitespace before a period
    p_str = re.sub(r'\s+\?', '?', p_str)  # Remove any whitespace before a question mark
    p_str = re.sub(r'\s+!', '!', p_str)  # Remove any whitespace before an exclamation point
    # Remove white space between an ' and an s if there is a white space after the s (i.e. possessive apostrophe) or is this is a punctuation mark {., !, ?, :}
    p_str = re.sub(r"'\s+s(\s|[.,!?;:])", r"'s\1", p_str)
    # Remove footnote numbers at end of a sentence. Check for a digit at the end and drop it
    # until there are no more digits or the sentence is now a valid end of a sentence.
    while p_str and p_str[-1].isdigit() and not is_sentence_end(p_str):
        p_str = p_str[:-1].strip()
    return p_str.strip()
