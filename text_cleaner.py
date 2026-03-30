import json
import ollama
from typing import Literal

ClassificationType = Literal['body', 'footnote', 'drop']

SYSTEM_PROMPT = """You are a text cleaning assistant for a book-to-audio conversion system.
You will be given a paragraph of text extracted from a PDF or EPUB book, along with the 
previous paragraph for context.

Your job is to:
1. Clean the text by fixing OCR errors, removing stray footnote markers (e.g. trailing numbers 
   like "word.1"), fixing word breaks (e.g. "hyphen- ated" should become "hyphenated"), and 
   correcting encoding artifacts.
2. Classify the paragraph as one of:
   - "body": main content of the book that should be read aloud
    - "footnote": footnote or endnote content. Key signals include:
         * The current paragraph starts with a number (e.g. "1 This ignores..." or "2 See also...")
         * The previous paragraph does not end with sentence-ending punctuation (. ? !), 
           suggesting the body text continues on the next page and this text is a footnote 
           inserted at the bottom of the page
         * Both signals together are a strong indicator of a footnote
    - "drop": content that should not be read aloud, such as:
         * Table of contents (lines with chapter names followed by page numbers, 
           often with dots or spaces between them, e.g. "Chapter 1 ... 1", "Introduction ... 5")
         * Index entries
         * Bibliography or references list
         * Publisher information or copyright notices
         * Page headers or page footers
         * Any other non-body content

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

    def clean(self, paragraph: str, previous_paragraph: str = "") -> tuple[str, ClassificationType]:
        """Clean and classify a paragraph of text.

        Args:
            paragraph: The paragraph text to clean and classify.
            previous_paragraph: The previous paragraph for context. Defaults to empty string.

        Returns:
            A tuple of (cleaned_text, classification) where classification
            is 'body', 'footnote', or 'drop'.

        Raises:
            ValueError: If the LLM returns a malformed response after all retries.
        """
        if previous_paragraph:
            user_content = f"Previous paragraph:\n{previous_paragraph}\n\nCurrent paragraph:\n{paragraph}"
        else:
            user_content = f"Current paragraph:\n{paragraph}"

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                response = ollama.chat(
                    model=self._model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_content}
                    ]
                )
                content: str = response['message']['content'].strip()
                print(f"LLM response: {repr(content)}")  # temporary debug
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
