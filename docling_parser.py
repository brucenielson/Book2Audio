from typing import List, Dict, Tuple, Optional, Union
import re
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import SectionHeaderItem, ListItem, TextItem, DocItem
from word_validator import word_validator
from docling_core.types.doc.document import ProvenanceItem


def is_section_header(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == "section_header"


def is_page_footer(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == "page_footer"


def is_page_header(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == "page_header"


def is_footnote(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == "footnote"


def is_list_item(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return False
    return text.label == "list_item"


def is_text_break(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return is_page_header(text) or is_section_header(text) or is_footnote(text)


def is_page_not_text(text: DocItem | None) -> bool:
    if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
        return True
    return text.label not in ["text", "list_item", "formula"]


def is_page_text(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
    return not is_page_not_text(text)


def is_ends_with_punctuation(text: str) -> bool:
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


def is_smaller_text(doc_item: DocItem, doc: DoclingDocument, threshold: float = 0.8) -> bool:
    """
    Determine if a DocItem's text is smaller than the average text size on its page.

    Parameters:
    - doc_item: The DocItem object containing provenance data with 'bbox'.
    - doc: The DoclingDocument containing all DocItems.
    - threshold: Ratio of the average text size to consider as 'smaller text'.

    Returns:
    - True if the DocItem's text is smaller than the average text size, False otherwise.
    """
    # Check if the DocItem has provenance data with a bounding box
    # noinspection PyTypeHints
    if hasattr(doc_item.prov[0], 'bbox'):
        # noinspection PyTypeHints
        bbox = doc_item.prov[0].bbox
    else:
        return False  # No bounding box available

    # Extract the bounding box coordinates
    x0: float = bbox.l
    y0: float = bbox.b
    x1: float = bbox.r
    y1: float = bbox.t

    # Calculate the area of the DocItem's bounding box
    doc_item_area: float = (x1 - x0) * (y1 - y0)

    # Filter doc_items that are on the same page
    # noinspection PyTypeHints
    same_page_items: List[DocItem] = [item for item in doc.texts if item.prov[0].page_no == doc_item.prov[0].page_no]

    # Calculate the average area of bounding boxes on the page
    # noinspection PyTypeHints
    total_area: float = sum(
        (item.prov[0].bbox.r - item.prov[0].bbox.l) * (item.prov[0].bbox.t - item.prov[0].bbox.b)
        for item in same_page_items if hasattr(item.prov[0], 'bbox')
    )
    # noinspection PyTypeHints
    num_items: int = sum(1 for item in same_page_items if hasattr(item.prov[0], 'bbox'))
    average_area: float = total_area / num_items if num_items > 0 else 0

    # Compare the DocItem's area to the average
    return doc_item_area < average_area * threshold


def is_too_short(doc_item: DocItem, threshold: int = 2) -> bool:
    return doc_item.label == "text" and len(doc_item.text) <= threshold


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


def get_next_text(texts: List[Union[SectionHeaderItem, ListItem, TextItem]], i: int) \
        -> Optional[Union[ListItem, TextItem]]:
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


def get_current_page(text: Union[SectionHeaderItem, ListItem, TextItem],
                     combined_paragraph: str,
                     current_page: Optional[int]) -> Optional[int]:
    # noinspection PyTypeHints
    return text.prov[0].page_no if current_page is None or combined_paragraph == "" else current_page


def should_skip_element(text: Union[SectionHeaderItem, ListItem, TextItem]) -> bool:
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


class DoclingParser:
    def __init__(self, doc: DoclingDocument,
                 meta_data: dict[str, str],
                 min_paragraph_size: int = 300,
                 start_page: Optional[int] = None,
                 end_page: Optional[int] = None,
                 double_notes: bool = False) -> None:
        self._doc: DoclingDocument = doc
        self._min_paragraph_size: int = min_paragraph_size
        self._docs_list: List[str] = []
        self._meta_list: List[Dict[str, str]] = []
        self._meta_data: dict[str, str] = meta_data
        self._start_page: Optional[int] = start_page
        self._end_page: Optional[int] = end_page
        self._double_notes: bool = double_notes

    def run(self, debug: bool = False) -> Tuple[List[str], List[Dict[str, str]]]:
        temp_docs: List[str] = []
        temp_meta: List[Dict[str, str]] = []
        combined_paragraph: str = ""
        i: int
        combined_chars: int = 0
        para_num: int = 0
        section_name: str = ""
        page_no: Optional[int] = None
        first_note: bool = False

        texts: List[DocItem] = self._get_processed_texts()

        for i, text in enumerate(texts):
            next_text: Optional[Union[ListItem, TextItem]] = get_next_text(texts, i)
            page_no = get_current_page(text, combined_paragraph, page_no)

            # Check if the current page is within the valid range
            if self._start_page is not None and page_no is not None and page_no < self._start_page:
                page_no = None
                continue
            if self._end_page is not None and page_no is not None and page_no > self._end_page:
                if self._double_notes and not first_note:
                    self._min_paragraph_size *= 2
                    first_note = True
                continue

            # Update section header if the element is a section header
            if is_section_header(text):
                section_name = text.text
                # Flush the current accumulated paragraph before the section header
                if combined_paragraph:
                    combined_paragraph = word_validator.combine_hyphenated_words(combined_paragraph)
                    para_num += 1
                    self._add_paragraph(combined_paragraph, para_num, section_name, page_no, temp_docs, temp_meta)
                    combined_paragraph, combined_chars = "", 0
                    page_no = None
                # Add the section header itself as its own paragraph
                header_str: str = clean_text(text.text)
                if header_str:
                    para_num += 1
                    self._add_paragraph(header_str, para_num, section_name, page_no, temp_docs, temp_meta)
                    page_no = None
                continue

            if should_skip_element(text):
                continue

            p_str: str = clean_text(text.text)
            p_str_chars: int = len(p_str)

            # If the paragraph does not end with final punctuation, accumulate it
            if not is_sentence_end(p_str):
                combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                combined_chars += p_str_chars
                continue

            # p_str ends with a sentence end; decide whether to process or accumulate it
            total_chars: int = combined_chars + p_str_chars
            if is_section_header(next_text):
                # Immediately process if the next text is a section header
                p_str = combine_paragraphs(combined_paragraph, p_str)
                combined_paragraph, combined_chars = "", 0
            elif total_chars < self._min_paragraph_size:
                # Not enough characters accumulated yet; decide based on next_text
                if next_text is None or (not is_page_text(next_text) and is_sentence_end(p_str)):
                    # End of document or next text item is not a text item and current paragraph ends with punctuation
                    # Process the paragraph and reset the accumulator even though this is a short paragraph
                    p_str = combine_paragraphs(combined_paragraph, p_str)
                    combined_paragraph, combined_chars = "", 0
                else:
                    # Combine with next paragraph
                    combined_paragraph = combine_paragraphs(combined_paragraph, p_str)
                    combined_chars = total_chars
                    continue
            else:
                # Sufficient characters: process the paragraph and reset the accumulator
                p_str = combine_paragraphs(combined_paragraph, p_str)
                combined_paragraph, combined_chars = "", 0

            p_str = word_validator.combine_hyphenated_words(p_str)
            if p_str:  # Only add non-empty content
                para_num += 1
                self._add_paragraph(p_str, para_num, section_name, page_no, temp_docs, temp_meta)
                page_no = None

        if debug:
            # Print the processed text to a file in the same directory as the document with the name of the document and _processed_texts.txt at the end
            output_path: str = "documents/" + self._doc.name + "_processed_texts.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for text in texts:
                    # noinspection PyTypeHints
                    f.write(f"{text.prov[0].page_no if text.prov else 'N/A'}: {text.label}: {text.text}\n")

            output_path = "documents/" + self._doc.name + "_processed_paragraphs.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for text in temp_docs:
                    f.write(text + "\n\n")

            return [], []  # Return empty lists if in debug mode after writing the processed texts to a file

        return temp_docs, temp_meta

    def _get_processed_texts(self) -> List[DocItem]:
        """
        Processes the document's text items page by page, separating regular content from notes
        (footnotes and bottom notes), and returns a list of DocItems with notes at the end.
        """
        regular_texts: List[DocItem] = []
        notes: List[DocItem] = []
        processed_pages: set[int] = set()  # Keep track of processed pages

        text_item: DocItem
        for text_item in self._doc.texts:
            page_number: int = text_item.prov[0].page_no

            if page_number not in processed_pages:
                # On new page, so get all items on the current page
                same_page_items: List[DocItem] = [
                    item for item in self._doc.texts if item.prov[0].page_no == page_number
                ]
                processed_pages.add(page_number)  # Mark the page as processed

            if is_too_short(text_item):
                continue
            elif is_footnote(text_item):
                notes.append(text_item)
            else:
                regular_texts.append(text_item)

        return regular_texts + notes

    def _add_paragraph(self, text: str, para_num: int, section: str,
                       page: Optional[int], docs: List[str], meta: List[Dict]) -> None:
        docs.append(text)
        meta.append({
            **self._meta_data,
            # "paragraph_#": str(para_num),
            "section_name": section,
            "page_#": str(page)
        })
