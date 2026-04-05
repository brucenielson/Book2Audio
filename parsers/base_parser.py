from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from text_cleaner import TextCleaner


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    # noinspection PyUnusedLocal
    @abstractmethod
    def __init__(self, source,
                 include_footnotes: bool = False,
                 meta_data: dict[str, str] | None = None,
                 min_paragraph_size: int = 0,
                 cleaner: TextCleaner | None = None) -> None:
        """Initialize the parser.

        Args:
            source: The document to parse. Accepts either a file path or a
                    preloaded document object. Supported types vary by subclass.
            include_footnotes: If True, footnote content is included in the
                               output alongside body text. Defaults to False.
            meta_data: Base metadata dict to include with every paragraph.
                       Defaults to None (empty metadata).
            min_paragraph_size: Minimum character count before a paragraph is
                                emitted. For audio output, 0 is a reasonable
                                default since short paragraphs are simply read
                                as brief pauses. Defaults to 0.
            cleaner: Optional TextCleaner for LLM-based cleaning and
                     classification. Defaults to None (rule-based only).
        """
        pass

    @abstractmethod
    def run(self, generate_text_file: bool = False) -> Tuple[List[str], List[Dict[str, str]]]:
        """Parse the document and return paragraphs and metadata.

        Returns:
            A tuple of (docs, meta) where docs is a list of paragraph strings
            and meta is a list of metadata dicts, one per paragraph.
        """
        pass
