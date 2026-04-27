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
                                 compute_single_line_height,
                                 compute_median_chars_per_line,
                                 is_small_text,
                                 is_single_line)
from utils.general_utils import is_sentence_end


@dataclasses.dataclass
class _FootnoteContext:
    """Transient classification state shared across the classification methods.

    Holds both the per-iteration loop state (which changes as we walk the
    document) and the document-level metrics (computed once before the loop).
    Bundled here so _is_footnote(), _is_running_head(), and _update_text_state()
    all receive what they need in one argument.
    """
    prev_text_candidate: bool       # last TEXT item was long with no sentence end
    prev_ends_mid_sentence: bool    # last body TEXT item ended with alpha or non-terminal punct
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
        if page_no is None:
            return True
        if self._start_page is not None and page_no < self._start_page:
            return False
        if self._end_page is not None and page_no > self._end_page:
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

    @staticmethod
    def _compute_boundary_indices(items: list[TextItem]) -> set[int]:
        """Return the indices of the first and last item on each page.

        Running page headers/footers mislabeled as section headers always appear
        at the first or last position on their page.

        Args:
            items: All text items in document order.

        Returns:
            A set of indices that are the first or last item on their page.
        """
        page_first: dict[int, int] = {}
        page_last: dict[int, int] = {}
        for i, item in enumerate(items):
            pno: int = item.prov[0].page_no
            if pno not in page_first:
                page_first[pno] = i
            page_last[pno] = i
        return set(page_first.values()) | set(page_last.values())

    @staticmethod
    def _is_running_head(i: int, text_item: TextItem,
                         boundary_indices: set[int], ctx: _FootnoteContext) -> bool:
        """Return True if this section header looks like a mislabeled running page header.

        A genuine running head is always: labeled SECTION_HEADER, at the top or
        bottom of a page, preceded by long mid-sentence body text, and single-line.
        All four conditions must hold before the item is suppressed.

        Args:
            i: Index of text_item in all_text_items.
            text_item: The item to evaluate.
            boundary_indices: Indices that are first or last on their page.
            ctx: Current classification context.

        Returns:
            True if the item should be suppressed as a running page header.
        """
        return (text_item.label == DocItemLabel.SECTION_HEADER
                and i in boundary_indices
                and ctx.prev_text_candidate
                and ctx.prev_ends_mid_sentence
                and is_single_line(text_item, ctx.single_line_height))

    def _update_text_state(self, text_item: TextItem, ctx: _FootnoteContext) -> None:
        """Update tracking state after an item is routed to regular body text.

        For TEXT items, refreshes prev_text_candidate (used by the footnote H1
        heuristic) and prev_ends_mid_sentence (used by the running-head guard).
        Non-TEXT items reset prev_ends_mid_sentence so subsequent section headers
        are not incorrectly suppressed.

        Args:
            text_item: The item just routed to regular_texts.
            ctx: The context to update in place.
        """
        if text_item.label == DocItemLabel.TEXT:
            ctx.prev_text_candidate = (len(text_item.text) >= self._short_text_threshold
                                       and not is_sentence_end(text_item.text))
            last_char: str = text_item.text.rstrip()[-1] if text_item.text.rstrip() else ''
            ctx.prev_ends_mid_sentence = last_char.isalpha() or last_char in (',', ':', ';')
        else:
            ctx.prev_ends_mid_sentence = False

    def _get_processed_texts(self) -> tuple[list[TextItem], list[TextItem]]:
        """Separate the document's text items into regular content and footnotes.

        Collects valid TextItems, computes document-level font-size baselines,
        then classifies each item using _is_footnote() and _is_running_head().

        Returns:
            A tuple of (regular_texts, notes) where each is a list of TextItems.
        """
        # Collect all valid TextItems. Page headers and footers are excluded
        # upfront — they are not body content and should never influence
        # classification or appear in the debug file.
        all_text_items: list[TextItem] = [
            item for item in self._doc.texts
            if not should_skip_element(item)
        ]
        single_line_height: float = compute_single_line_height(self._doc)
        median_chars_per_line: float = compute_median_chars_per_line(
            all_text_items, single_line_height, min_charspan=self._short_text_threshold
        )

        boundary_indices: set[int] = self._compute_boundary_indices(all_text_items)

        regular_texts: list[TextItem] = []
        notes: list[TextItem] = []
        current_page: int | None = None
        ctx: _FootnoteContext = _FootnoteContext(
            prev_text_candidate=False,
            prev_ends_mid_sentence=False,
            text_seen_this_page=False,
            found_note_this_page=False,
            single_line_height=single_line_height,
            median_chars_per_line=median_chars_per_line,
        )

        for i, text_item in enumerate(all_text_items):
            page_number: int = text_item.prov[0].page_no

            if page_number != current_page:
                ctx.text_seen_this_page = False
                ctx.found_note_this_page = False
                current_page = page_number

            if is_too_short(text_item):
                continue

            if DoclingParser._is_running_head(i, text_item, boundary_indices, ctx):
                continue

            went_to_notes: bool = self._is_footnote(text_item, ctx)
            if went_to_notes:
                ctx.found_note_this_page = True
                notes.append(text_item)
            else:
                regular_texts.append(text_item)
                self._update_text_state(text_item, ctx)

            if not went_to_notes and text_item.label == DocItemLabel.TEXT:
                ctx.text_seen_this_page = True

        return regular_texts, notes
