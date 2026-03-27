from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple

class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def __init__(self, source, meta_data: dict[str, str],
                 min_paragraph_size: int = 300) -> None:
        pass

    @abstractmethod
    def run(self, generate_text_file: bool = False) -> Tuple[List[str], List[Dict[str, str]]]:
        """Parse the document and return paragraphs and metadata.

        Returns:
            A tuple of (docs, meta) where docs is a list of paragraph strings
            and meta is a list of metadata dicts, one per paragraph.
        """
        pass
