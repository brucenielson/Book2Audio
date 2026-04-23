from __future__ import annotations

import re
import time
from pathlib import Path

from text_chunk import RawChunk, ParsedChunk
from word_validator import word_validator
from utils.general_utils import is_sentence_end, build_paragraph, clean_text
from utils.logging_utils import vprint
from text_cleaner import TextCleaner


def _all_words_valid(text: str, verbose: bool = False) -> bool:
    """Return True if every token in text is a known English word.

    Only common sentence punctuation is stripped before checking (commas,
    periods, colons, etc.). Tokens containing digits or any other non-letter
    characters are treated as potential artifacts and return False immediately.

    Args:
        text: The paragraph text to validate.
        verbose: If True, prints each token that fails validation. Defaults to False.
    """
    for token in text.split():
        stripped = re.sub(r"[,;:.!?()'\"—–]", '', token.lower())
        if not stripped or not word_validator.is_valid_word(stripped):
            vprint(verbose, f"  [FAIL TOKEN] {token!r} -> {stripped!r}")
            return False
    return True


class TextProcessor:
    """Processes a list of RawChunks into cleaned, accumulated ParsedChunks.

    Handles paragraph accumulation, section header flushing, and hyphen
    combining. Optionally writes the final paragraphs to a text file.

    Attributes:
        _min_paragraph_size: Minimum character count before a paragraph is emitted.
        _include_footnotes: If True, footnote chunks are included in the output.
        _cleaner: Optional TextCleaner for LLM-based cleaning and classification.
    """

    def __init__(self, min_paragraph_size: int = 0,
                 include_footnotes: bool = False,
                 cleaner: str | TextCleaner | None = None,
                 verbose: bool = False) -> None:
        """Initialise TextProcessor.

        Args:
            min_paragraph_size: Minimum character count before a paragraph is emitted.
            include_footnotes: If True, footnote chunks are included in the output.
            cleaner: Optional LLM model name (str), TextCleaner instance, or None.
                     A string is interpreted as an Ollama model name and used to
                     create a TextCleaner automatically. Defaults to None.
            verbose: If True, prints per-paragraph skip/LLM decisions and timing
                     summary. Defaults to False.
        """
        self._min_paragraph_size: int = min_paragraph_size
        self._include_footnotes: bool = include_footnotes
        self._verbose: bool = verbose
        if isinstance(cleaner, str):
            self._cleaner: TextCleaner | None = TextCleaner(model=cleaner)
        else:
            self._cleaner = cleaner  # TextCleaner instance or None
        self._paragraph: list[str] = []
        self._section_name: str = ""
        self._para_num: int = 0
        self._result: list[ParsedChunk] = []
        self._page_contexts: dict[str, str] = {}
        self._t_validation: float = 0.0
        self._t_llm: float = 0.0
        self._n_skipped: int = 0
        self._n_llm_calls: int = 0

    @property
    def _combined_count(self) -> int:
        return sum(len(p) for p in self._paragraph)

    def _init_state(self) -> None:
        """Reset all processing state before a new run."""
        self._paragraph = []
        self._section_name = ""
        self._para_num = 0
        self._result = []
        self._page_contexts = {}
        self._t_validation = 0.0
        self._t_llm = 0.0
        self._n_skipped = 0
        self._n_llm_calls = 0

    def _clear_state(self) -> None:
        """Clear processing state after a run to free memory."""
        self._paragraph = []
        self._section_name = ""
        self._para_num = 0
        self._result = []
        self._page_contexts = {}

    def process(self, chunks: list[RawChunk],
                output_path: Path | None = None,
                generate_text_file: bool = False) -> list[ParsedChunk]:
        """Process a list of RawChunks into ParsedChunks.

        Args:
            chunks: The list of RawChunks to process.
            output_path: Base path for output files. Required if generate_text_file is True.
            generate_text_file: If True, writes processed paragraphs to a text file.

        Returns:
            A list of ParsedChunks ready for audio output.
        """
        self._init_state()

        # Eagerly warm up word_validator so NLTK resources are loaded before
        # processing begins rather than on the first call mid-paragraph.
        if self._cleaner is not None:
            word_validator.is_valid_word('warm')

        # Clean all chunks upfront
        for chunk in chunks:
            chunk.text = clean_text(chunk.text, remove_footnotes=True)

        # Build page context strings for LLM-based cleaning
        if self._cleaner is not None:
            self._page_contexts = self._build_page_contexts(chunks)

        for i, chunk in enumerate(chunks):
            next_chunk: RawChunk | None = chunks[i + 1] if i < len(chunks) - 1 else None

            if chunk.is_section_header:
                self._handle_section_header(chunk)
                continue

            if chunk.is_page_header or chunk.is_page_footer:
                continue

            if chunk.is_footnote and not self._include_footnotes:
                continue

            self._process_chunk(chunk, next_chunk)

        # Flush any remaining accumulated paragraph
        if self._paragraph:
            self._flush_paragraph({})

        if generate_text_file and output_path is not None:
            self._save_paragraphs_file(output_path)

        if self._cleaner and self._verbose:
            total = self._t_validation + self._t_llm
            vprint(self._verbose,
                   f"\n[TIMING] validation={self._t_validation:.2f}s ({self._n_skipped} skipped) | "
                   f"llm={self._t_llm:.2f}s ({self._n_llm_calls} calls) | "
                   f"total_timed={total:.2f}s")

        result = self._result
        self._clear_state()
        return result

    def _should_accumulate(self, p_str: str, next_chunk: RawChunk | None) -> bool:
        """Return True if p_str should be accumulated rather than emitted.

        There are two reasons to accumulate:
        1. The paragraph is incomplete — it doesn't end with sentence-ending punctuation,
           so we must wait for more text before emitting.
        2. The paragraph is complete but too short — we haven't reached min_paragraph_size
           yet and there is more text coming, so we combine with the next chunk.
        """
        if next_chunk is None:
            # End of document — emit whatever we have
            return False

        # Incomplete paragraph — must accumulate regardless of size
        if not is_sentence_end(p_str):
            return True

        # Complete paragraph — check if we should still accumulate due to size
        total_char_count: int = self._combined_count + len(p_str)

        if next_chunk.is_section_header:
            # Next element is a section header — emit now to avoid crossing a section boundary
            return False
        if not next_chunk.is_body_text:
            # Next element is not body text — emit now
            return False
        if total_char_count >= self._min_paragraph_size:
            # Reached minimum size — emit now
            return False

        # Paragraph is complete but short and more text is coming — accumulate
        return True

    def _build_meta(self, meta: dict[str, str]) -> dict[str, str]:
        """Build the metadata dict for an emitted ParsedChunk.

        Args:
            meta: The base metadata from the source chunk.

        Returns:
            A new metadata dict that includes paragraph number and section name.
        """
        result = {
            **meta,
            "paragraph_#": str(self._para_num),
        }
        if self._section_name:
            result["section_name"] = self._section_name
        return result

    def _handle_section_header(self, chunk: RawChunk) -> None:
        """Flush any accumulated paragraph and emit the section header.

        Args:
            chunk: The section header RawChunk.
        """
        self._section_name = chunk.text
        if self._paragraph:
            self._flush_paragraph(chunk.meta)
        if chunk.text:
            self._para_num += 1
            self._result.append(ParsedChunk(
                text=chunk.text,
                meta=self._build_meta(chunk.meta),
                label=chunk.label
            ))

    def _build_page_contexts(self, chunks: list[RawChunk]) -> dict[str, str]:
        """Build a mapping of page number to full page text.

        Args:
            chunks: All RawChunks for the document.

        Returns:
            Dict mapping page_# values to concatenated page text.
        """
        page_texts: dict[str, list[str]] = {}
        for chunk in chunks:
            page_num = chunk.meta.get('page_#', '')
            if page_num:
                page_texts.setdefault(page_num, []).append(chunk.text)
        return {page: '\n\n'.join(texts) for page, texts in page_texts.items()}

    def _build_paragraph(self) -> str:
        """Build a single paragraph string from the accumulated chunks.

        Returns:
            A single paragraph string.
        """
        return build_paragraph(self._paragraph)

    def _flush_paragraph(self, meta: dict[str, str], label: str = 'text') -> None:
        """Flush the accumulated paragraph as a ParsedChunk.

        If a cleaner is configured, calls the LLM to clean and classify the
        paragraph. 'drop' paragraphs are discarded; 'footnote' paragraphs are
        discarded unless include_footnotes is True.

        Args:
            meta: Metadata to attach to the flushed paragraph.
            label: The label for the emitted ParsedChunk.
        """
        p_str: str = self._build_paragraph()

        if self._cleaner:
            t0 = time.perf_counter()
            _skip = _all_words_valid(p_str, verbose=self._verbose)
            self._t_validation += time.perf_counter() - t0

            vprint(self._verbose, f"{'[SKIP]' if _skip else '[LLM ] '} {p_str[:100]!r}")
            if not _skip:
                self._n_llm_calls += 1
                page_context = self._page_contexts.get(meta.get('page_#', ''), '')
                t1 = time.perf_counter()
                p_str, classification = self._cleaner.clean(p_str, page_context=page_context)
                self._t_llm += time.perf_counter() - t1
                if classification == 'drop':
                    self._paragraph = []
                    return
                if classification == 'footnote':
                    if not self._include_footnotes:
                        self._paragraph = []
                        return
                    label = 'footnote'
            else:
                self._n_skipped += 1

        p_str = word_validator.combine_hyphenated_words(p_str)
        if p_str:
            self._para_num += 1
            self._result.append(ParsedChunk(
                text=p_str,
                meta=self._build_meta(meta),
                label=label
            ))
        self._paragraph = []

    def _process_chunk(self, chunk: RawChunk, next_chunk: RawChunk | None) -> None:
        """Process a single body text chunk.

        Args:
            chunk: The current RawChunk to process.
            next_chunk: The next RawChunk in the list, or None if at end.
        """
        p_str: str = chunk.text

        if self._should_accumulate(p_str, next_chunk):
            self._paragraph.append(p_str)
            return

        # Ready to emit — accumulate and flush
        self._paragraph.append(p_str)
        self._flush_paragraph(chunk.meta, label=chunk.label)

    def _save_paragraphs_file(self, output_path: Path) -> None:
        """Write processed paragraphs to a text file.

        Args:
            output_path: Base path for the output file. The file will be named
                         <output_path>_processed_paragraphs.txt.
        """
        with open(f"{output_path}_processed_paragraphs.txt", "w", encoding="utf-8") as f:
            for chunk in self._result:
                f.write(chunk.text + "\n\n")
