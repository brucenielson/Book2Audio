"""Docling-based PDF parser for Book2Audio."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import DocItem, TextItem, DocItemLabel

from text_chunk import RawChunk, ParsedChunk
from text_processor import TextProcessor
from text_cleaner import TextCleaner
from parsers.base_parser import BaseParser
from utils.docling_utils import (is_footnote,
                                 should_skip_element,
                                 is_too_short,
                                 get_current_page,
                                 load_as_document,
                                 is_text_bearing,
                                 compute_single_line_height,
                                 compute_median_chars_per_line,
                                 is_small_text)
from utils.general_utils import is_sentence_end


@dataclasses.dataclass
class _FootnoteContext:
    """Transient classification state passed to the footnote detector.

    Holds both the per-iteration loop state (which changes as we walk the
    document) and the document-level metrics (computed once before the loop).
    Bundled here so _is_footnote() receives everything it needs in one argument
    rather than five.
    """
    prev_text_candidate: bool       # last TEXT item was long with no sentence end
    text_seen_this_page: bool       # body text has been seen on the current page
    found_note_this_page: bool      # a footnote has been seen on the current page
    single_line_height: float       # median height of one line (from headers/footers)
    median_chars_per_line: float    # median chars-per-estimated-line for the document


class DoclingParser(BaseParser):
    """Parser for PDF documents using the Docling library."""

    def __init__(self, source: str | Path | DoclingDocument,
                 include_footnotes: bool = False,
                 meta_data: dict[str, str] | None = None,
                 min_paragraph_size: int = 5,
                 start_page: int | None = None,
                 end_page: int | None = None,
                 llm_cleaner: str | TextCleaner | None = None,
                 min_footnote_chars: int = 100) -> None:
        """Initialise DoclingParser.

        Args:
            source: Path to the PDF file, or a preloaded DoclingDocument instance.
                    If a file path is provided, the document is loaded and cached
                    as a JSON file alongside the source for faster future runs.
            include_footnotes: If True, footnote content is included in the
                               output alongside body text. Defaults to False.
            meta_data: Base metadata dict to include with every paragraph.
                       Defaults to None (empty metadata).
            min_paragraph_size: Minimum character count before a paragraph is
                                emitted. For audio output, 0 is a reasonable
                                default since short paragraphs are simply read
                                as brief pauses. Defaults to 0.
            start_page: Optional first page to include. Pages before this are
                        skipped. Defaults to None (start from beginning).
            end_page: Optional last page to include. Pages after this are
                      skipped. Defaults to None (read to end).
            llm_cleaner: Optional TextCleaner for LLM-based cleaning and classification.
                     Defaults to None (rule-based cleaning only).
            min_footnote_chars: Minimum character count applied across unlabelled
                                footnote detection. Controls the minimum length of
                                a preceding body-text item for the sentence-end
                                heuristic to fire, the minimum length of a
                                candidate item for the small-text heuristic, and
                                the minimum charspan used when computing the
                                document's median characters-per-line baseline.
                                Defaults to 100.
        """
        if isinstance(source, DoclingDocument):
            self._doc: DoclingDocument = source
            self._file_path: Path | None = None
        else:
            self._file_path = Path(source)
            self._doc = load_as_document(self._file_path)

        self._min_paragraph_size: int = min_paragraph_size
        self._meta_data: dict[str, str] = meta_data or {}
        self._start_page: int | None = start_page
        self._end_page: int | None = end_page
        self._include_notes: bool = include_footnotes
        self._cleaner: str | TextCleaner | None = llm_cleaner
        self._short_text_threshold: int = min_footnote_chars

    def _is_in_page_range(self, page_no: int | None) -> bool:
        """Check whether a page number falls within the configured page range.

        Args:
            page_no: The page number to check, or None.

        Returns:
            True if the page is within [start_page, end_page], False otherwise.
        """
        if self._start_page is not None and page_no is not None and page_no < self._start_page:
            return False
        if self._end_page is not None and page_no is not None and page_no > self._end_page:
            return False
        return True

    def run(self, generate_text_file: bool = False) -> tuple[list[str], list[dict[str, str]]]:
        """Parse the document and return paragraphs and metadata.

        Args:
            generate_text_file: If True, saves processed text and paragraph files
                                 alongside the source document.

        Returns:
            A tuple of (docs, meta) where docs is a list of paragraph strings
            and meta is a list of metadata dicts, one per paragraph.
        """
        raw_chunks: list[RawChunk] = self._extract_chunks()

        output_path: Path | None = None
        if generate_text_file and self._file_path is not None:
            output_path = self._file_path.parent / self._doc.name

        processor: TextProcessor = TextProcessor(
            min_paragraph_size=self._min_paragraph_size,
            include_footnotes=self._include_notes,
            cleaner=self._cleaner
        )

        parsed_chunks: list[ParsedChunk] = processor.process(
            chunks=raw_chunks,
            output_path=output_path,
            generate_text_file=generate_text_file
        )

        if generate_text_file and self._file_path is not None:
            regular_texts, notes = self._extract_all_texts()
            self._save_text_files(regular_texts, notes)

        docs: list[str] = [chunk.text for chunk in parsed_chunks]
        meta: list[dict[str, str]] = [chunk.meta for chunk in parsed_chunks]
        return docs, meta

    def _extract_chunks(self) -> list[RawChunk]:
        """Extract raw chunks from the document.

        Returns:
            A list of RawChunks extracted from the document.
        """
        chunks: list[RawChunk] = []
        page_no: int | None = None

        regular_texts, notes = self._get_processed_texts()
        texts: list[DocItem] = regular_texts + (notes if self._include_notes else [])

        for i, text in enumerate(texts):
            if not is_text_bearing(text):
                continue

            page_no = get_current_page(text, "", page_no)

            if not self._is_in_page_range(page_no):
                continue

            if should_skip_element(text):
                continue

            meta: dict[str, str] = {
                **self._meta_data,
                "section_name": "",
                "page_#": str(page_no)
            }

            assert isinstance(text, TextItem)
            p_str: str = text.text
            if not p_str:
                continue

            chunks.append(RawChunk(
                text=p_str,
                meta=meta,
                label=text.label
            ))

        return chunks

    def _extract_all_texts(self) -> tuple[list[TextItem], list[TextItem]]:
        """Return all classified DocItems for debug file writing.

        Returns:
            A tuple of (regular_texts, notes) containing all classified items.
            The caller decides whether to include notes.
        """
        return self._get_processed_texts()

    def _save_text_files(self, regular_texts: list[DocItem], notes: list[DocItem]) -> None:
        """Write per-item debug text to a file alongside the source document.

        Body text is written first, followed by a separator line, then footnotes.

        Args:
            regular_texts: Body text DocItems to write.
            notes: Footnote DocItems to write after the separator.

        Raises:
            ValueError: If no file path is available (document was passed directly).
        """
        if self._file_path is None:
            raise ValueError(
                "Cannot save text files when DoclingDocument was passed directly — no file path available.")
        base_path: Path = self._file_path.parent / self._doc.name

        with open(f"{base_path}_processed_texts.txt", "w", encoding="utf-8") as f:
            for text in regular_texts:
                text_content: str = text.text if is_text_bearing(text) else 'N/A' # noqa
                # noinspection PyTypeHints
                f.write(f"{text.prov[0].page_no if text.prov else 'N/A'}: {text.label}: {text_content}\n")
            f.write("--- FOOTNOTES ---\n")
            for text in notes:
                text_content = text.text if is_text_bearing(text) else 'N/A' # noqa
                # noinspection PyTypeHints
                f.write(f"{text.prov[0].page_no if text.prov else 'N/A'}: {text.label}: {text_content}\n")

    def _is_footnote(self, text_item: TextItem, ctx: _FootnoteContext) -> bool:
        """Return True if text_item should be classified as a footnote.

        Checks Docling's own FOOTNOTE label first, then applies three
        unlabelled-footnote heuristics for TEXT items that start with a digit:

        1. Sentence-end heuristic: the preceding TEXT item was substantial and
           ended mid-sentence, making a digit-start continuation a near-certain
           footnote reference.
        2. Small-text heuristic: the item is noticeably smaller than the document's
           body text (more chars per estimated line than the median).
        3. Propagation heuristic: a footnote has already been seen on this page,
           so subsequent digit+alpha items are treated as continuations.

        Args:
            text_item: The item to classify.
            ctx: Current classification context (page state and document metrics).

        Returns:
            True if the item is or should be classified as a footnote.
        """
        if text_item.label == DocItemLabel.FOOTNOTE:
            return True
        if not (text_item.label == DocItemLabel.TEXT
                and text_item.text
                and text_item.text[0].isdigit()):
            return False
        has_alpha: bool = any(c.isalpha() for c in text_item.text)
        return (
            # H1: follows mid-sentence body text; alpha check excludes index entries like "183-84"
            (has_alpha and ctx.prev_text_candidate)
            # H2: small font, preceded by body text on this page
            or (len(text_item.text) >= self._short_text_threshold
                and ctx.text_seen_this_page
                and is_small_text(text_item, ctx.single_line_height, ctx.median_chars_per_line))
            # H3: propagation — footnote already seen on this page
            or (has_alpha and ctx.found_note_this_page)
        )

    def _get_processed_texts(self) -> tuple[list[TextItem], list[TextItem]]:
        """Separate the document's text items into regular content and footnotes.

        Uses two passes: the first computes the median bbox.height/charspan_length
        ratio across all TextItems (a proxy for font size); the second classifies
        each item using _is_footnote().

        Returns:
            A tuple of (regular_texts, notes) where each is a list of TextItems.
        """
        # First pass: collect all valid TextItems and compute font-size baselines
        all_text_items: list[TextItem] = [
            item for item in self._doc.texts if is_text_bearing(item)
        ]
        single_line_height: float = compute_single_line_height(self._doc)
        median_chars_per_line: float = compute_median_chars_per_line(
            all_text_items, single_line_height, min_charspan=self._short_text_threshold
        )

        regular_texts: list[TextItem] = []
        notes: list[TextItem] = []
        carry_over: bool = False             # prev_text_candidate intentionally kept from previous page
        current_page: int | None = None
        ctx: _FootnoteContext = _FootnoteContext(
            prev_text_candidate=False,
            text_seen_this_page=False,
            found_note_this_page=False,
            single_line_height=single_line_height,
            median_chars_per_line=median_chars_per_line,
        )

        text_item: TextItem
        for text_item in all_text_items:
            page_number: int = text_item.prov[0].page_no

            if page_number != current_page:
                ctx.text_seen_this_page = False
                ctx.found_note_this_page = False
                if ctx.prev_text_candidate:
                    carry_over = True   # keep prev_text_candidate visible for exactly one TEXT item
                else:
                    ctx.prev_text_candidate = False  # redundant but explicit reset for clarity
                current_page = page_number

            if is_too_short(text_item):
                continue
            elif self._is_footnote(text_item, ctx):
                text_item.label = DocItemLabel.FOOTNOTE
                ctx.found_note_this_page = True
                notes.append(text_item)
            else:
                regular_texts.append(text_item)
                if text_item.label == DocItemLabel.TEXT:
                    ctx.prev_text_candidate = (len(text_item.text) >= self._short_text_threshold
                                               and not is_sentence_end(text_item.text))

            if carry_over and text_item.label == DocItemLabel.TEXT:
                # label is still TEXT → item went to else (notes branches mutate to FOOTNOTE)
                # → prev_text_candidate was already updated by the else branch above
                carry_over = False

            if text_item.label == DocItemLabel.TEXT:
                ctx.text_seen_this_page = True

        return regular_texts, notes
