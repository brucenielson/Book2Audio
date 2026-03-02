from typing import List, Dict, Tuple
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import DocItem, SectionHeaderItem, ListItem, TextItem
from word_validator import word_validator
from utils.docling_utils import (is_section_header,
                                 is_footnote,
                                 is_page_text,
                                 is_sentence_end,
                                 should_skip_element,
                                 is_too_short,
                                 combine_paragraphs,
                                 get_next_text,
                                 get_current_page,
                                 clean_text)


class DoclingParser:
    def __init__(self, doc: DoclingDocument,
                 meta_data: dict[str, str],
                 min_paragraph_size: int = 300,
                 start_page: int | None = None,
                 end_page: int | None = None,
                 include_notes: bool = True) -> None:
        self._doc: DoclingDocument = doc
        self._min_paragraph_size: int = min_paragraph_size
        self._meta_data: dict[str, str] = meta_data
        self._start_page: int | None = start_page
        self._end_page: int | None = end_page
        self._include_notes: bool = include_notes

    def run(self, debug: bool = False) -> Tuple[List[str], List[Dict[str, str]]]:
        temp_docs: List[str] = []
        temp_meta: List[Dict[str, str]] = []
        combined_paragraph: str = ""
        i: int
        combined_chars: int = 0
        para_num: int = 0
        section_name: str = ""
        page_no: int | None = None

        regular_texts, notes = self._get_processed_texts()
        texts: List[DocItem] = regular_texts + (notes if self._include_notes else [])

        for i, text in enumerate(texts):
            # We only deal with SectionHeaderItem, ListItem, and TextItem; skip anything else
            if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
                continue

            next_text: DocItem | None = get_next_text(texts, i)
            page_no = get_current_page(text, combined_paragraph, page_no)

            # Check if the current page is within the valid range
            if self._start_page is not None and page_no is not None and page_no < self._start_page:
                page_no = None
                continue
            if self._end_page is not None and page_no is not None and page_no > self._end_page:
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

            if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
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
                    text_content: str = text.text if isinstance(text,
                                                                (SectionHeaderItem, ListItem, TextItem)) else 'N/A'
                    f.write(f"{text.prov[0].page_no if text.prov else 'N/A'}: {text.label}: {text_content}\n")

            output_path = "documents/" + self._doc.name + "_processed_paragraphs.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                for text in temp_docs:
                    f.write(text + "\n\n")

            return [], []  # Return empty lists if in debug mode after writing the processed texts to a file

        return temp_docs, temp_meta

    def _get_processed_texts(self) -> Tuple[List[DocItem], List[DocItem]]:
        """
        Processes the document's text items, separating regular content from notes
        (footnotes), and returns them as separate lists.
        """
        regular_texts: List[DocItem] = []
        notes: List[DocItem] = []
        processed_pages: set[int] = set()  # Keep track of processed pages

        text_item: DocItem
        for text_item in self._doc.texts:
            # noinspection PyTypeHints
            page_number: int = text_item.prov[0].page_no

            if page_number not in processed_pages:
                # On new page, so get all items on the current page
                # noinspection PyTypeHints
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

        return regular_texts, notes

    def _add_paragraph(self, text: str, para_num: int, section: str,
                       page: int | None, docs: List[str], meta: List[Dict]) -> None:
        docs.append(text)
        meta.append({
            **self._meta_data,
            "paragraph_#": str(para_num),
            "section_name": section,
            "page_#": str(page)
        })
