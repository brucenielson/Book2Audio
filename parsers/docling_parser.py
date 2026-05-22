"""Docling-based PDF parser for Book2Audio."""

from __future__ import annotations

import dataclasses
import itertools
import re
from pathlib import Path

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem, DocItemLabel

from text_chunk import RawChunk, ParsedChunk
from text_processor import TextProcessor
from text_cleaner import TextCleaner
from parsers.base_parser import BaseParser
from utils.docling_utils import (is_footnote,
                                 is_text_bearing,
                                 is_too_short,
                                 should_skip_element,
                                 load_as_document,
                                 compute_single_line_height,
                                 compute_median_chars_per_line,
                                 compute_body_line_height,
                                 is_small_text,
                                 is_single_line,
                                 get_pdf_page_labels,
                                 is_front_matter,
                                 calibrate_header_top_y,
                                 compute_median_page_height,
                                 is_math_heavy)
from utils.general_utils import is_sentence_end


@dataclasses.dataclass
class _FootnoteContext:
    """Transient classification state shared across the classification methods.

    Holds both the per-iteration loop state (which changes as we walk the
    document) and the document-level metrics (computed once before the loop).
    Bundled here so _is_footnote(), _is_page_header(), and _update_text_state()
    all receive what they need in one argument.
    """
    prev_text_candidate: bool       # last TEXT item was long with no sentence end (H1 footnote gate)
    text_seen_this_page: bool       # body text has been seen on the current page
    found_note_this_page: bool      # a footnote has been seen on the current page
    single_line_height: float       # median height of one line (from headers/footers)
    median_chars_per_line: float    # median chars-per-estimated-line for the document
    header_top_y: float | None      # median bbox.t of validated PAGE_HEADER items, or None
    median_page_height: float       # median page height across the document
    body_line_height: float         # median bbox.height of single-line body TEXT items
    in_notes_section: bool          # True once a "Notes" / "Endnotes" section header is seen


class DoclingParser(BaseParser):
    """Parser for PDF documents using the Docling library."""

    _SKIP_LABELS: frozenset[str] = frozenset({'footnote', 'page_header', 'too_short'})

    def __init__(self, source: str | Path | DoclingDocument,
                 include_footnotes: bool = False,
                 meta_data: dict[str, str] | None = None,
                 min_paragraph_size: int = 5,
                 start_page: int | None = None,
                 end_page: int | None = None,
                 llm_cleaner: str | TextCleaner | None = None,
                 min_footnote_chars: int = 100,
                 verbose: bool = False,
                 page_labels: dict[int, str] | None = None,
                 skip_front_matter: bool = False,
                 skip_index: bool = False) -> None:
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
            verbose: If True, prints progress messages during conversion. Defaults to False.
            page_labels: Optional pre-loaded mapping from physical page index (0-based,
                         as returned by pypdfium2) to the printed label string.  When
                         None and a file path is provided, labels are loaded automatically
                         from the PDF.  Pass an explicit dict (including {}) to override.
            skip_front_matter: If True, pages whose PDF label is a Roman numeral are
                                excluded from output.  Requires page labels to be
                                available (loaded automatically from the PDF when a file
                                path is given).  Defaults to False.
            skip_index: If True, the back-matter index section is detected automatically
                        and excluded from output along with all pages that follow it.
                        Detection uses _find_index_start_page(); see that method for the
                        exact rules.  Defaults to False.
        """
        if isinstance(source, DoclingDocument):
            self._doc: DoclingDocument = source
            self._file_path: Path | None = None
            self._page_labels: dict[int, str] = page_labels if page_labels is not None else {}
        else:
            self._file_path = Path(source)
            self._doc = load_as_document(self._file_path)
            self._page_labels = (page_labels if page_labels is not None
                                 else get_pdf_page_labels(self._file_path))

        self._min_paragraph_size: int = min_paragraph_size
        self._meta_data: dict[str, str] = meta_data or {}
        self._start_page: int | None = start_page
        self._end_page: int | None = end_page
        self._include_notes: bool = include_footnotes
        self._cleaner: str | TextCleaner | None = llm_cleaner
        self._short_text_threshold: int = min_footnote_chars
        self._verbose: bool = verbose
        self._skip_front_matter: bool = skip_front_matter
        self._skip_index: bool = skip_index

    def _page_label_for(self, page_no: int) -> str:
        """Return the printed page label for a Docling page number.

        Docling uses 1-based page numbers; pypdfium2 uses 0-based indices.
        Falls back to the physical page number string if no label is available.

        Args:
            page_no: Docling's 1-based physical page number.

        Returns:
            The PDF label string (e.g. 'i', 'xl', '1', '368'), or str(page_no)
            if no label table was loaded.
        """
        label = self._page_labels.get(page_no - 1)
        return label or str(page_no)

    # Regex for the SECTION_HEADER signal: matches 'index', 'indexes', or 'indices'
    # as a complete word (word-boundary anchored, case-insensitive).
    # 'Indexical' is intentionally excluded — after 'index' the next character 'i'
    # is a word character, so the word boundary \\b does not fire there.
    _INDEX_SECTION_RE: re.Pattern[str] = re.compile(
        r'\bindex(?:es)?\b|\bindices\b', re.IGNORECASE
    )

    def _find_index_start_page(self) -> int | None:
        """Find the physical page number where the back-matter index section begins.

        Scans self._doc.texts for two types of signals that indicate an index:

        Signal 1 — PAGE_HEADER:
            The running page header text, after stripping ALL whitespace, contains
            the substring 'index' (case-insensitive).  This handles OCR-spaced
            titles such as 'I N DEX OF SUBJ ECTS', which normalise to
            'indexofsubjects' and clearly contain 'index'.  A plain 'Index of
            Names' header normalises to 'indexofnames' and also matches.

        Signal 2 — SECTION_HEADER:
            The section heading text matches the regex \\bindex(?:es)?\\b|\\bindices\\b,
            which covers 'Index', 'Indexes', and 'Indices' as whole words.
            The word-boundary anchor prevents 'Indexical' from matching — after
            'index' in 'Indexical' comes the letter 'i', a word character, so
            no word boundary fires there.
            Unlike Signal 1, this check is done on the original text (not
            whitespace-stripped), because a genuine section heading is unlikely
            to have OCR spacing artifacts between individual letters.

        Position gate:
            Both signals are subject to a position gate that rejects matches in
            the first 70% of the document.  The total page count is estimated
            from the highest page_no seen across all items in self._doc.texts.
            This prevents false positives from chapter titles that contain the
            word 'index' early in the body (e.g. a philosophy chapter on
            'Indexical Reference').  Only items in the last 30% of the book
            can trigger index detection.

        Returns:
            The minimum physical page_no where a signal fires, or None if no
            index section is detectable.
        """
        # First pass: determine the total page count so we can apply the position gate.
        max_page: int = 0
        for item in self._doc.texts:
            if item.prov:
                max_page = max(max_page, item.prov[0].page_no)
        if max_page == 0:
            return None

        # Any signal on a page at or before this threshold is ignored.
        position_threshold: int = int(max_page * 0.70)

        index_page: int | None = None

        for item in self._doc.texts:
            if not item.prov:
                continue
            page_no: int = item.prov[0].page_no
            if page_no <= position_threshold:
                continue  # position gate: ignore the first 70% of the book

            if item.label == DocItemLabel.PAGE_HEADER:
                # Signal 1: strip all whitespace and look for 'index' as a substring.
                # Handles both clean titles ('Index') and OCR-spaced ones
                # ('I N DEX OF SUBJ ECTS' → 'indexofsubjects').
                normalized: str = re.sub(r'\s+', '', item.text.lower())
                if 'index' in normalized:
                    if index_page is None or page_no < index_page:
                        index_page = page_no

            elif item.label == DocItemLabel.SECTION_HEADER:
                # Signal 2: word-boundary match on the original text.
                # Matches 'Index', 'Indices', 'Indexes', 'Indice' as whole words.
                # Does NOT match 'Indexical' (word boundary after 'x' blocks it).
                if self._INDEX_SECTION_RE.search(item.text):
                    if index_page is None or page_no < index_page:
                        index_page = page_no

        return index_page

    def _format_page(self, page_no: int) -> str:
        """Format a page reference for display in text output files.

        When the PDF label matches the physical page number, returns just the
        number.  When they differ (e.g. Roman numeral front matter, or a book
        whose PDF labels don't start at 1), returns '[Page <label> / Page <physical>]'.

        Args:
            page_no: Docling's 1-based physical page number.

        Returns:
            A display string such as '42', 'i', or '[Page 1 / Page 41]'.
        """
        label = self._page_label_for(page_no)
        physical = str(page_no)
        if label == physical:
            return label
        return f'[Page {label} / Page {physical}]'

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

    def run(self, generate_text_file: bool = False,
            annotate_reclassifications: bool = False) -> tuple[list[str], list[dict[str, str]]]:
        """Parse the document and return paragraphs and metadata.

        Args:
            generate_text_file: If True, saves processed text and paragraph files
                                 alongside the source document.
            annotate_reclassifications: If True, items whose label was changed by our
                                        classification show 'original → new' in
                                        _processed_texts.txt. Defaults to False so
                                        existing canonical files are unaffected; flip to
                                        True once canonicals are regenerated.

        Returns:
            A tuple of (docs, meta) where docs is a list of paragraph strings
            and meta is a list of metadata dicts, one per paragraph.
        """
        classified: list[tuple[TextItem, str]] = self._get_processed_texts()
        regular_texts: list[TextItem] = [item for item, label in classified
                                         if label not in DoclingParser._SKIP_LABELS]
        notes: list[TextItem] = [item for item, label in classified if label == 'footnote']
        raw_chunks: list[RawChunk] = self._extract_chunks(regular_texts, notes)

        output_path: Path | None = None
        if generate_text_file and self._file_path is not None:
            output_path = self._file_path.parent / self._doc.name

        processor: TextProcessor = TextProcessor(
            min_paragraph_size=self._min_paragraph_size,
            include_footnotes=self._include_notes,
            cleaner=self._cleaner,
            verbose=self._verbose
        )

        parsed_chunks: list[ParsedChunk] = processor.process(
            chunks=raw_chunks,
            output_path=output_path,
            generate_text_file=generate_text_file
        )

        if generate_text_file and self._file_path is not None:
            self._save_text_files(classified, annotate_reclassifications=annotate_reclassifications)

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

        # Compute the index start page once up front (None when skip_index is False
        # or when no index section is detected).
        index_start: int | None = self._find_index_start_page() if self._skip_index else None

        chunks: list[RawChunk] = []
        for text in all_items:
            page_no: int = text.prov[0].page_no
            if not self._is_in_page_range(page_no):
                continue
            label: str = self._page_label_for(page_no)
            if self._skip_front_matter and is_front_matter(label):
                continue
            if index_start is not None and page_no >= index_start:
                continue  # skip the index section and all back matter that follows
            chunks.append(RawChunk(
                text=text.text,
                meta={**self._meta_data, "section_name": "",
                      "page_#": label, "physical_page_#": str(page_no)},
                label=text.label
            ))

        return chunks

    def _save_text_files(self, classified: list[tuple[TextItem, str]],
                         annotate_reclassifications: bool = False) -> None:
        """Write per-item debug text to a file alongside the source document.

        All items are written in document order. When annotate_reclassifications is True,
        items whose label was changed show 'original_label → new_label:'; otherwise just
        the original label is shown.

        Args:
            classified: List of (item, final_label) pairs in document order,
                        as returned by _get_processed_texts().
            annotate_reclassifications: If True, show original → new label for reclassified items.

        Raises:
            ValueError: If no file path is available (document was passed directly).
        """
        if self._file_path is None:
            raise ValueError(
                "Cannot save text files when DoclingDocument was passed directly — no file path available.")
        base_path: Path = self._file_path.parent / self._doc.name

        with open(f"{base_path}_processed_texts.txt", "w", encoding="utf-8") as f:
            for text_item, final_label in classified:
                page = self._format_page(text_item.prov[0].page_no) if text_item.prov else 'N/A'
                original_label = str(text_item.label)
                if annotate_reclassifications and final_label != original_label:
                    f.write(f"{page}: {original_label} → {final_label}: {text_item.text}\n")
                else:
                    f.write(f"{page}: {original_label}: {text_item.text}\n")

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
        if not (text_item.label in (DocItemLabel.TEXT, DocItemLabel.SECTION_HEADER)
                and text_item.text):
            return False
        # H3: propagation — once a footnote has been seen on this page, all subsequent
        # TEXT items are footnote continuations regardless of first character.
        # SECTION_HEADERs are excluded: they represent chapter/section titles and
        # are not swept up by propagation.
        if text_item.label == DocItemLabel.TEXT and ctx.found_note_this_page:
            return True
        # Endnote path: in a dedicated notes/endnotes section (back of book), any
        # digit-start TEXT item with alpha content is an endnote. No font-size or
        # page-position requirements — endnote pages may use the same font as body text.
        if (ctx.in_notes_section
                and text_item.label == DocItemLabel.TEXT
                and text_item.text[0].isdigit()
                and any(c.isalpha() for c in text_item.text)):
            return True
        # H4: 1–2 digits immediately against an uppercase letter or opening punctuation
        # (e.g. "3See", "14Cf", "3[See", "8(See", "13'That").  The tight juxtaposition
        # of a digit marker and a word/punctuation is almost never body text.
        # Requires body text to have been seen first on the page.
        # Uppercase avoids ordinals like "1st". TEXT label only — SECTION_HEADERs are
        # chapter/section titles and must not be swept up here.
        if (ctx.text_seen_this_page
                and text_item.label == DocItemLabel.TEXT
                and re.match(r'^\d{1,2}[A-Z\[(\'\"]', text_item.text)):
            return True
        # H1: digit-start item with alpha content that immediately follows a mid-sentence
        # body paragraph and sits in the lower half of the page.  No font-size requirement —
        # footnotes in narrow columns may be typeset at the same size as body text but will
        # always appear below the main text block.
        if (text_item.label == DocItemLabel.TEXT
                and text_item.text[0].isdigit()
                and any(c.isalpha() for c in text_item.text)
                and ctx.prev_text_candidate
                and ctx.median_page_height > 0
                and text_item.prov
                and text_item.prov[0].bbox is not None
                and text_item.prov[0].bbox.t < ctx.median_page_height * 0.5):
            return True
        # Gate: footnotes are always smaller than body text, and cannot appear before
        # body text has been seen on the page.
        if not (ctx.text_seen_this_page
                and is_small_text(text_item, ctx.single_line_height,
                                  ctx.median_chars_per_line,
                                  body_line_height=ctx.body_line_height)):
            return False
        # Gate: first character must be a digit.
        first: str = text_item.text[0]
        if not first.isdigit():
            return False
        # Numbered list items are not footnotes — "1. Introduction", "2. Method", etc.
        if re.match(r'^\d+\.\s', text_item.text):
            return False
        # H1 (small-text path): small text following a mid-sentence body paragraph.
        # alpha check excludes pure index entries like "183-84".
        has_alpha: bool = any(c.isalpha() for c in text_item.text)
        if has_alpha and ctx.prev_text_candidate:
            return True
        # Lower half of page: any digit-start item in small text at the bottom is a footnote.
        if (ctx.median_page_height > 0
                and text_item.prov
                and text_item.prov[0].bbox is not None
                and text_item.prov[0].bbox.t < ctx.median_page_height * 0.5):
            return True
        return False

    @staticmethod
    def _is_page_header(text_item: TextItem, ctx: _FootnoteContext) -> bool:
        """Return True if this section header looks like a mislabeled running page header.

        Uses two paths depending on whether validated PAGE_HEADER items were found
        during calibration:

        Path A (ctx.header_top_y is not None): the item must be within
        ±single_line_height of the reference y-coordinate established from real
        PAGE_HEADER items.

        Path B (ctx.header_top_y is None): no reference is available, so the item
        must appear in the top 15% of the page.

        In both paths the item must be a single-line SECTION_HEADER with no prior
        body text on the same page.

        Args:
            text_item: The item to evaluate.
            ctx: Current classification context including calibrated header position.

        Returns:
            True if the item should be suppressed as a running page header.
        """
        if text_item.label != DocItemLabel.SECTION_HEADER:
            return False
        if not is_single_line(text_item, ctx.single_line_height):
            return False
        if not text_item.prov:
            return False
        bbox = text_item.prov[0].bbox
        if bbox is None:
            return False

        if ctx.header_top_y is not None:
            # Path A: validated reference — must be within one line-height of reference y
            return abs(bbox.t - ctx.header_top_y) <= ctx.single_line_height
        else:
            # Path B: no reference — must be in the top 15% of the page.
            # Docling PDFs use BOTTOMLEFT coordinates: bbox.t increases going up, so
            # "near the top" means a large bbox.t (close to page height). We compute
            # distance from the top as (page_height - bbox.t) and check if that is
            # less than 15% of page height.
            if ctx.median_page_height <= 0:
                return False
            return (ctx.median_page_height - bbox.t) / ctx.median_page_height < 0.15

    def _update_text_state(self, text_item: TextItem, ctx: _FootnoteContext) -> None:
        """Update tracking state after an item is routed to regular body text.

        For TEXT items, refreshes prev_text_candidate (used by the footnote H1
        heuristic to detect unlabelled footnotes that follow mid-sentence body text).
        A sentence-ending text item or a text item ending with a colon clears the
        candidate flag, since a following digit-start item is not a plausible footnote.

        Args:
            text_item: The item just routed to regular_texts.
            ctx: The context to update in place.
        """
        if text_item.label == DocItemLabel.TEXT:
            text_stripped = text_item.text.rstrip()
            ends_sentence = is_sentence_end(text_stripped) or text_stripped.endswith(':')
            ctx.prev_text_candidate = (len(text_item.text) >= self._short_text_threshold
                                       and not ends_sentence)

    def _get_processed_texts(self) -> list[tuple[TextItem, str]]:
        """Classify the document's text items and return them in document order.

        Collects valid TextItems, computes document-level font-size baselines,
        then classifies each item using _is_footnote() and _is_page_header().

        Returns:
            A list of (item, label) pairs in document order. Label is one of:
            the original Docling label string (body text / section headers),
            'footnote', 'page_header', or 'too_short' for suppressed items.
        """
        # Collect all valid TextItems. Page headers and footers are excluded.
        all_text_items: list[TextItem] = [
            item for item in self._doc.texts
            if not should_skip_element(item)
        ]

        # Sort within each page by bbox.t descending so items are processed in
        # physical top-to-bottom order regardless of Docling's emission order.
        # Docling sometimes emits footnotes (physically at the bottom) before body
        # text (physically near the top) on the same page; without sorting, H3
        # propagation would sweep body text that follows a footnote in emission
        # order but is physically above it.
        #
        # Exception: pages with two columns have items spread across a wide
        # horizontal range.  A pure Y-sort would interleave left and right
        # columns, producing nonsense reading order.  If the bbox.l spread on a
        # page exceeds 100 pts we treat it as multi-column and leave Docling's
        # emission order intact for that page.
        def _page_no(item: TextItem) -> int:
            return item.prov[0].page_no if item.prov else 999_999

        # Stable sort by page number first so itertools.groupby sees contiguous pages.
        all_text_items.sort(key=_page_no)

        reordered: list[TextItem] = []
        for _pno, page_iter in itertools.groupby(all_text_items, key=_page_no):
            page_items = list(page_iter)
            # Only TEXT items contribute to the column-spread check.
            # Section headers are often centered (large l) and formulas are
            # indented, so including them produces false positives on
            # single-column pages like p.293 of Realism and the Aim of Science.
            l_values: list[float] = []
            for it in page_items:
                if it.label != DocItemLabel.TEXT:
                    continue
                if it.prov and it.prov[0].bbox is not None:
                    l = getattr(it.prov[0].bbox, 'l', None)
                    if isinstance(l, (int, float)):
                        l_values.append(float(l))
            if l_values and (max(l_values) - min(l_values)) > 100.0:
                # Multi-column page: preserve Docling's emission order.
                reordered.extend(page_items)
            else:
                # Single-column page: sort top-to-bottom by bbox.t descending.
                page_items.sort(key=lambda it: (
                    -it.prov[0].bbox.t
                    if it.prov and it.prov[0].bbox is not None
                    else float('inf')
                ))
                reordered.extend(page_items)

        all_text_items = reordered
        single_line_height: float = compute_single_line_height(self._doc)
        median_chars_per_line: float = compute_median_chars_per_line(
            all_text_items, single_line_height, min_charspan=self._short_text_threshold
        )

        header_top_y: float | None = calibrate_header_top_y(self._doc)
        median_page_height: float = compute_median_page_height(self._doc)
        body_line_height: float = compute_body_line_height(
            list(self._doc.texts), single_line_height
        )

        classified: list[tuple[TextItem, str]] = []
        current_page: int | None = None
        ctx: _FootnoteContext = _FootnoteContext(
            prev_text_candidate=False,
            text_seen_this_page=False,
            found_note_this_page=False,
            single_line_height=single_line_height,
            median_chars_per_line=median_chars_per_line,
            header_top_y=header_top_y,
            median_page_height=median_page_height,
            body_line_height=body_line_height,
            in_notes_section=False,
        )

        for text_item in all_text_items:
            page_number: int = text_item.prov[0].page_no

            if page_number != current_page:
                ctx.text_seen_this_page = False
                ctx.found_note_this_page = False
                current_page = page_number

            # Detect start of endnotes section — once set, stays True for the rest of the book.
            if (not ctx.in_notes_section
                    and text_item.label == DocItemLabel.SECTION_HEADER
                    and text_item.text
                    and re.match(r'^\s*(notes?|endnotes?)\b', text_item.text, re.IGNORECASE)):
                ctx.in_notes_section = True

            if is_too_short(text_item):
                classified.append((text_item, 'too_short'))
                continue

            if DoclingParser._is_page_header(text_item, ctx):
                classified.append((text_item, 'page_header'))
                continue

            went_to_notes: bool = self._is_footnote(text_item, ctx)
            if went_to_notes:
                ctx.found_note_this_page = True
                classified.append((text_item, 'footnote'))
            else:
                label: str = str(text_item.label)
                if text_item.label == DocItemLabel.TEXT and is_math_heavy(text_item.text):
                    label = 'formula'
                classified.append((text_item, label))
                self._update_text_state(text_item, ctx)

            if not went_to_notes and text_item.label == DocItemLabel.TEXT:
                ctx.text_seen_this_page = True

        return classified
