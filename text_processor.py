from pathlib import Path
from typing import List, Dict
from text_chunk import RawChunk, ParsedChunk
from word_validator import word_validator
from utils.general_utils import is_sentence_end, combine_paragraphs, clean_text


class TextProcessor:
    """Processes a list of RawChunks into cleaned, accumulated ParsedChunks.

    Handles paragraph accumulation, section header flushing, and hyphen
    combining. Optionally writes the final paragraphs to a text file.

    Attributes:
        _min_paragraph_size: Minimum character count before a paragraph is emitted.
        _include_footnotes: If True, footnote chunks are included in the output.
    """

    def __init__(self, min_paragraph_size: int = 300, include_footnotes: bool = False) -> None:
        """Initialise TextProcessor.

        Args:
            min_paragraph_size: Minimum character count before a paragraph is emitted.
            include_footnotes: If True, footnote chunks are included in the output.
        """
        self._min_paragraph_size: int = min_paragraph_size
        self._include_footnotes: bool = include_footnotes
        self._combined_paragraph: str = ""
        self._combined_count: int = 0
        self._section_name: str = ""
        self._para_num: int = 0
        self._result: List[ParsedChunk] = []

    def _init_state(self) -> None:
        self._combined_paragraph = ""
        self._combined_count = 0
        self._section_name = ""
        self._para_num = 0
        self._result = []

    def _clear_state(self) -> None:
        self._combined_paragraph = ""
        self._combined_count = 0
        self._section_name = ""
        self._para_num = 0
        self._result = []

    def process(self, chunks: List[RawChunk],
                output_path: Path | None = None,
                generate_text_file: bool = False) -> List[ParsedChunk]:
        """Process a list of RawChunks into ParsedChunks.

        Args:
            chunks: The list of RawChunks to process.
            output_path: Base path for output files. Required if generate_text_file is True.
            generate_text_file: If True, writes processed paragraphs to a text file.

        Returns:
            A list of ParsedChunks ready for audio output.
        """
        self._init_state()

        for i, chunk in enumerate(chunks):
            next_chunk: RawChunk | None = chunks[i + 1] if i < len(chunks) - 1 else None

            # Do initial cleaning
            chunk.text = clean_text(chunk.text)

            if chunk.is_section_header:
                self._handle_section_header(chunk)
                continue

            if chunk.is_page_header or chunk.is_page_footer:
                continue

            if chunk.is_footnote and not self._include_footnotes:
                continue

            self._process_chunk(chunk, next_chunk)

        # Flush any remaining accumulated paragraph
        if self._combined_paragraph:
            self._flush_paragraph({})

        if generate_text_file and output_path is not None:
            self._save_paragraphs_file(output_path)

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

    def _build_meta(self, meta: Dict[str, str]) -> Dict[str, str]:
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
        if self._combined_paragraph:
            self._flush_paragraph(chunk.meta)
        if chunk.text:
            self._para_num += 1
            self._result.append(ParsedChunk(
                text=chunk.text,
                meta=self._build_meta(chunk.meta),
                label=chunk.label
            ))

    def _flush_paragraph(self, meta: Dict[str, str], label: str = 'text') -> None:
        """Flush the accumulated paragraph as a ParsedChunk.

        Args:
            meta: Metadata to attach to the flushed paragraph.
            label: The label for the emitted ParsedChunk.
        """
        p_str: str = word_validator.combine_hyphenated_words(self._combined_paragraph)
        if p_str:
            self._para_num += 1
            self._result.append(ParsedChunk(
                text=p_str,
                meta=self._build_meta(meta),
                label=label
            ))
        self._combined_paragraph, self._combined_count = "", 0

    def _process_chunk(self, chunk: RawChunk, next_chunk: RawChunk | None) -> None:
        """Process a single body text chunk.

        Args:
            chunk: The current RawChunk to process.
            next_chunk: The next RawChunk in the list, or None if at end.
        """
        p_str: str = chunk.text

        if self._should_accumulate(p_str, next_chunk):
            self._combined_paragraph = combine_paragraphs(self._combined_paragraph, p_str)
            self._combined_count += len(p_str)
            return

        # Ready to emit — accumulate and flush
        self._combined_paragraph = combine_paragraphs(self._combined_paragraph, p_str)
        self._combined_count += len(p_str)
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
