import json
import ollama
from typing import Literal

ClassificationType = Literal['body', 'footnote', 'drop']

SYSTEM_PROMPT = """You are a text cleaning assistant for a book-to-audio conversion system.
You will be given a paragraph of text extracted from a PDF or EPUB book.

Your job is to:
1. Clean the text by fixing OCR errors, removing stray footnote markers (e.g. trailing numbers like "word.1"), 
   fixing word breaks (e.g. "hyphen-\nated" should become "hyphenated"), and correcting encoding artifacts.
2. Classify the paragraph as one of:
   - "body": main content of the book that should be read aloud
   - "footnote": footnote or endnote content (notes referenced from the body text, whether at the 
     bottom of a page or end of a chapter/book)
   - "drop": content that should not be read aloud, such as table of contents, index, bibliography, 
     publisher information, copyright notices, page headers, page footers, or other non-body content

Rules:
- Do NOT reword, paraphrase, or alter the actual prose
- Only fix genuine errors — if text looks correct, leave it unchanged
- Respond ONLY with a JSON object, no preamble or markdown backticks

Response format:
{
    "cleaned": "the cleaned paragraph text",
    "classification": "body" | "footnote" | "drop"
}"""


class TextCleaner:
    """Cleans and classifies paragraph text using a local LLM via Ollama.

    Attributes:
        _model: The Ollama model to use for cleaning.
        _max_retries: Maximum number of retries if the response is malformed.
    """

    def __init__(self, model: str = 'llama3.1:8b', max_retries: int = 3) -> None:
        """Initialise TextCleaner.

        Args:
            model: The Ollama model to use. Defaults to 'llama3.1:8b'.
            max_retries: Maximum number of retries on malformed responses. Defaults to 3.
        """
        self._model: str = model
        self._max_retries: int = max_retries

    def clean(self, paragraph: str) -> tuple[str, ClassificationType]:
        """Clean and classify a paragraph of text.

        Args:
            paragraph: The paragraph text to clean and classify.

        Returns:
            A tuple of (cleaned_text, classification) where classification
            is 'body', 'footnote', or 'drop'.

        Raises:
            ValueError: If the LLM returns a malformed response after all retries.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = ollama.chat(
                    model=self._model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': paragraph}
                    ]
                )
                content: str = response['message']['content'].strip()
                parsed = json.loads(content)

                cleaned: str = parsed['cleaned']
                classification: str = parsed['classification']

                if classification not in ('body', 'footnote', 'drop'):
                    raise ValueError(f"Invalid classification: '{classification}'")

                return cleaned, classification

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                last_error = e
                continue

        raise ValueError(f"Failed to get valid response after {self._max_retries} attempts. "
                         f"Last error: {last_error}")