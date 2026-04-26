"""Docling-based PDF parser for Book2Audio."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem, DocItemLabel

from text_chunk import RawChunk, ParsedChunk
from text_processor import TextProcessor
from text_cleaner import TextCleaner
from parsers.base_parser import BaseParser
from utils.docling_utils import (is_footnote,
                                 is_too_short,
                                 should_skip_element,
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
        regular_texts, notes = self._get_processed_texts()
        raw_chunks: list[RawChunk] = self._extract_chunks(regular_texts, notes)

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
            self._save_text_files(regular_texts, notes)

        docs: list[str] = [chunk.text for chunk in parsed_chunks]
        meta: list[dict[str, str]] = [chunk.meta for chunk in parsed_chunks]
        return docs, meta

    def _extract_chunks(self, regular_texts: list[TextItem],
                        notes: list[TextItem]) -> list[RawChunk]:
        """Build RawChunks from pre-classified text items, filtered to the page range.

        Args:
            regular_texts: Body text items from _get_processed_texts().
            notes: Footnote items from _get_processed_texts().

        Returns:
            A list of RawChunks ready for the text processor.
        """
        all_items: list[TextItem] = regular_texts + (notes if self._include_notes else [])

        chunks: list[RawChunk] = []
        for text in all_items:
            page_no: int = text.prov[0].page_no
            if not self._is_in_page_range(page_no):
                continue
            chunks.append(RawChunk(
                text=text.text,
                meta={**self._meta_data, "section_name": "", "page_#": str(page_no)},
                label=text.label
            ))

        return chunks

    def _save_text_files(self, regular_texts: list[TextItem], notes: list[TextItem]) -> None:
        """Write per-item debug text to a file alongside the source document.

        Body text is written first, followed by a separator line, then footnotes.
        Notes items are always written with the 'footnote' label regardless of their
        original Docling label, since reclassified items are not mutated.

        Args:
            regular_texts: Body text items from _get_processed_texts().
            notes: Footnote items from _get_processed_texts().

        Raises:
            ValueError: If no file path is available (document was passed directly).
        """
        if self._file_path is None:
            raise ValueError(
                "Cannot save text files when DoclingDocument was passed directly — no file path available.")
        base_path: Path = self._file_path.parent / self._doc.name

        with open(f"{base_path}_processed_texts.txt", "w", encoding="utf-8") as f:
            for text in regular_texts:
                f.write(f"{text.prov[0].page_no if text.prov else 'N/A'}: {text.label}: {text.text}\n")
            f.write("--- FOOTNOTES ---\n")
            for text in notes:
                f.write(f"{text.prov[0].page_no if text.prov else 'N/A'}: {text.label}: {text.text}\n")

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
        if is_footnote(text_item):
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
        # First pass: collect all valid TextItems and compute font-size baselines.
        # Page headers and footers are excluded upfront — they are not body content
        # and should never influence classification or appear in the debug file.
        all_text_items: list[TextItem] = [
            item for item in self._doc.texts
            if is_text_bearing(item) and not should_skip_element(item)
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

            went_to_notes: bool = self._is_footnote(text_item, ctx)
            if went_to_notes:
                ctx.found_note_this_page = True
                notes.append(text_item)
            else:
                regular_texts.append(text_item)
                if text_item.label == DocItemLabel.TEXT:
                    ctx.prev_text_candidate = (len(text_item.text) >= self._short_text_threshold
                                               and not is_sentence_end(text_item.text))

            if carry_over and not went_to_notes and text_item.label == DocItemLabel.TEXT:
                carry_over = False  # consumed by the first body TEXT item on the new page

            if not went_to_notes and text_item.label == DocItemLabel.TEXT:
                ctx.text_seen_this_page = True

        return regular_texts, notes
