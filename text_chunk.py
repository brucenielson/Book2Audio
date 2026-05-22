from __future__ import annotations

from dataclasses import dataclass, field

LABEL_FOOTNOTE: str = 'footnote'
LABEL_FORMULA: str = 'formula'


@dataclass
class TextChunk:
    """Base class for text chunks at any stage of the processing pipeline.

    Attributes:
        text: The text content of the chunk.
        meta: Metadata dict associated with the chunk.
        label: A label identifying the type of content, using DocItemLabel
               values for PDF (e.g. 'section_header', 'text', 'footnote')
               or HTML tag names for EPUB (e.g. 'h1', 'h2').
    """
    text: str
    meta: dict[str, str] = field(default_factory=dict)
    label: str = ""

    @property
    def is_section_header(self) -> bool:
        """True if this chunk is a section header or title."""
        return self.label in ('section_header', 'title', 'h1', 'h2', 'h3', 'h4', 'h5')

    @property
    def is_footnote(self) -> bool:
        """True if this chunk is a footnote."""
        return self.label == LABEL_FOOTNOTE

    @property
    def is_page_header(self) -> bool:
        """True if this chunk is a page header."""
        return self.label == 'page_header'

    @property
    def is_page_footer(self) -> bool:
        """True if this chunk is a page footer."""
        return self.label == 'page_footer'

    @property
    def is_formula(self) -> bool:
        """True if this chunk is a mathematical or logical formula."""
        return self.label == LABEL_FORMULA

    @property
    def is_too_short(self) -> bool:
        """True if this chunk was suppressed as too short to be meaningful."""
        return self.label == 'too_short'

    @property
    def is_body_text(self) -> bool:
        """True if this chunk is body text suitable for audio output."""
        return self.label in ('text', 'list_item', 'formula', 'paragraph')


@dataclass
class RawChunk(TextChunk):
    """A raw text chunk as extracted directly from a source document.

    Produced by DoclingParser or EpubParser before any accumulation,
    combining, or LLM-based cleaning has been applied.

    Attributes:
        original_label: The pre-reclassification label, set only when the
                        parser changed the Docling-assigned label (e.g. 'text'
                        → 'formula', 'section_header' → 'page_header').
                        Empty string when no reclassification occurred.
    """
    original_label: str = ""


@dataclass
class ParsedChunk(TextChunk):
    """A fully processed text chunk ready for audio output.

    Produced by TextProcessor after accumulation, combining, and
    cleaning have been applied to one or more RawChunks.
    """
    pass
