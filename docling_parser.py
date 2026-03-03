from pathlib import Path
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
                                 clean_text,
                                 load_as_document)


class DoclingParser:
    def __init__(self, source: str | Path | DoclingDocument,
                 meta_data: dict[str, str],
                 min_paragraph_size: int = 300,
                 start_page: int | None = None,
                 end_page: int | None = None,
                 include_notes: bool = True) -> None:
        if isinstance(source, DoclingDocument):
            self._doc: DoclingDocument = source
            self._file_path: Path | None = None
        else:
            self._file_path = Path(source)
            self._doc = load_as_document(self._file_path)

        self._min_paragraph_size: int = min_paragraph_size
        self._meta_data: dict[str, str] = meta_data
        self._start_page: int | None = start_page
        self._end_page: int | None = end_page
        self._include_notes: bool = include_notes

        # Run state — initialised in _init_run_state, cleared in _clear_run_state
        self._combined_paragraph: str = ""
        self._combined_count: int = 0
        self._para_num: int = 0
        self._section_name: str = ""
        self._page_no: int | None = None
        self._temp_docs: List[str] = []
        self._temp_meta: List[Dict[str, str]] = []

    def _init_run_state(self) -> None:
        self._combined_paragraph = ""
        self._combined_count = 0
        self._para_num = 0
        self._section_name = ""
        self._page_no = None
        self._temp_docs = []
        self._temp_meta = []

    def _clear_run_state(self) -> None:
        self._combined_paragraph = ""
        self._combined_count = 0
        self._para_num = 0
        self._section_name = ""
        self._page_no = None
        self._temp_docs = []
        self._temp_meta = []

    def _is_in_page_range(self) -> bool:
        if self._start_page is not None and self._page_no is not None and self._page_no < self._start_page:
            return False
        if self._end_page is not None and self._page_no is not None and self._page_no > self._end_page:
            return False
        return True

    def run(self, generate_text_file: bool = False) -> Tuple[List[str], List[Dict[str, str]]]:
        self._init_run_state()

        regular_texts, notes = self._get_processed_texts()
        texts: List[DocItem] = regular_texts + (notes if self._include_notes else [])

        for i, text in enumerate(texts):
            # We only deal with SectionHeaderItem, ListItem, and TextItem; skip anything else
            if not isinstance(text, (SectionHeaderItem, ListItem, TextItem)):
                continue

            next_text: DocItem | None = get_next_text(texts, i)
            self._page_no = get_current_page(text, self._combined_paragraph, self._page_no)

            if not self._is_in_page_range():
                self._page_no = None
                continue

            # Update section header if the element is a section header
            if is_section_header(text):
                self._handle_section_header(text)
                continue

            if should_skip_element(text):
                continue

            self._process_text_element(text, next_text)

        if generate_text_file:
            self._save_text_files(texts)

        result_docs, result_meta = self._temp_docs, self._temp_meta
        self._clear_run_state()
        return result_docs, result_meta

    def _save_text_files(self, texts: List[DocItem]) -> None:
        if self._file_path is None:
            raise ValueError(
                "Cannot save text files when DoclingDocument was passed directly — no file path available.")
        base_path: Path = self._file_path.parent / self._doc.name

        with open(f"{base_path}_processed_texts.txt", "w", encoding="utf-8") as f:
            for text in texts:
                text_content: str = text.text if isinstance(text, (SectionHeaderItem, ListItem, TextItem)) else 'N/A'
                # noinspection PyTypeHints
                f.write(f"{text.prov[0].page_no if text.prov else 'N/A'}: {text.label}: {text_content}\n")

        with open(f"{base_path}_processed_paragraphs.txt", "w", encoding="utf-8") as f:
            for text in self._temp_docs:
                f.write(text + "\n\n")

    def _handle_section_header(self, text: SectionHeaderItem | ListItem | TextItem) -> None:
        self._section_name = text.text
        # Flush the current accumulated paragraph before the section header
        if self._combined_paragraph:
            self._flush_paragraph()
        # Add the section header itself as its own paragraph
        header_str: str = clean_text(text.text)
        if header_str:
            self._para_num += 1
            self._add_paragraph(header_str, self._para_num, self._section_name, self._page_no,
                                 self._temp_docs, self._temp_meta)
            self._page_no = None

    def _flush_paragraph(self) -> None:
        self._combined_paragraph = word_validator.combine_hyphenated_words(self._combined_paragraph)
        self._para_num += 1
        self._add_paragraph(self._combined_paragraph, self._para_num, self._section_name, self._page_no,
                             self._temp_docs, self._temp_meta)
        self._combined_paragraph, self._combined_count = "", 0
        self._page_no = None

    def _should_accumulate(self, total_char_count: int, next_text: DocItem | None) -> bool:
        """Return True if the current paragraph should be accumulated rather than emitted."""
        if is_section_header(next_text):
            # Immediately process if the next text is a section header
            return False
        if total_char_count >= self._min_paragraph_size:
            # Too many characters accumulated, so accumulate no more
            return False
        if next_text is None:
            # End of document, so don't accumulate further
            return False
        if not is_page_text(next_text):
            # Next text is not page text, so don't accumulate
            return False
        return True

    def _process_text_element(self, text: SectionHeaderItem | ListItem | TextItem,
                              next_text: DocItem | None) -> None:
        p_str: str = clean_text(text.text)
        p_str_count: int = len(p_str)

        # If the paragraph does not end with final punctuation, accumulate it
        if not is_sentence_end(p_str):
            self._combined_paragraph = combine_paragraphs(self._combined_paragraph, p_str)
            self._combined_count += p_str_count
            return

        total_char_count: int = self._combined_count + p_str_count

        if self._should_accumulate(total_char_count, next_text):
            # Combine with next paragraph
            self._combined_paragraph = combine_paragraphs(self._combined_paragraph, p_str)
            self._combined_count = total_char_count
            return

        # Ready to emit — combine with any accumulated text and output
        p_str = combine_paragraphs(self._combined_paragraph, p_str)
        self._combined_paragraph, self._combined_count = "", 0

        p_str = word_validator.combine_hyphenated_words(p_str)
        if p_str:  # Only add non-empty content
            self._para_num += 1
            self._add_paragraph(p_str, self._para_num, self._section_name, self._page_no,
                                self._temp_docs, self._temp_meta)
            self._page_no = None

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
