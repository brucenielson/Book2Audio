"""LLM-based text cleaner and classifier for book-to-audio conversion."""

from __future__ import annotations

import difflib
import json
from typing import Literal, TypeAlias

import ollama

from utils.logging_utils import vprint
from word_validator import word_validator

ClassificationType: TypeAlias = Literal['body', 'footnote', 'drop']

# noinspection SpellCheckingInspection
SYSTEM_PROMPT: str = """You are a text cleaning assistant for a book-to-audio conversion system.
You will be given a paragraph of text extracted from a PDF or EPUB book, along with the
full text of the page it came from for context.

Your job is to:
1. Clean the text by fixing OCR errors, removing stray footnote markers (e.g. trailing numbers
   like "word.1"), fixing word breaks (e.g. "hyphen- ated" should become "hyphenated"), and
   correcting encoding artifacts.
2. Classify the paragraph as one of:
   - "body": main content of the book that should be read aloud
   - "footnote": footnote or endnote content. Key signals include:
         * The current paragraph starts with a number (e.g. "1 This ignores..." or "2 See also...")
         * The page context shows the paragraph appears at the bottom of the page after body text,
           which is where footnotes are typically placed
         * A corresponding footnote marker (e.g. a superscript or trailing number) appears in the
           body text earlier on the page
         * All of these signals together are a strong indicator of a footnote
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
- Do NOT change capitalization of words
- Do NOT drop any words from the original text unless they are clear OCR artifacts
- Only fix genuine errors — if text looks correct, leave it unchanged
- Respond ONLY with a JSON object, no preamble or markdown backticks

Response format:
{
    "cleaned": "the cleaned paragraph text",
    "classification": "body" | "footnote" | "drop"
}"""


# noinspection SpellCheckingInspection
_DROP_HINTS: tuple[str, ...] = (
    'index', 'bibliograph', 'reference', 'encyclop', 'glossar',
    'appendix', 'contents', 'header', 'footer', 'caption', 'table',
)
_FOOTNOTE_HINTS: tuple[str, ...] = ('footnote', 'endnote', 'note')
_BODY_HINTS: tuple[str, ...] = ('body', 'main', 'prose', 'content', 'paragraph', 'text')


def _coerce_classification(raw: str) -> ClassificationType | None:
    """Map a hallucinated classification label to the nearest valid one, or None."""
    lowered = raw.lower()
    if any(h in lowered for h in _FOOTNOTE_HINTS):
        return 'footnote'
    if any(h in lowered for h in _BODY_HINTS):
        return 'body'
    if any(h in lowered for h in _DROP_HINTS):
        return 'drop'
    return None


def _has_suspicious_substitutions(original: str, cleaned: str) -> bool:
    """Return True if the LLM made a suspicious word substitution.

    Legitimate fixes replace invalid words (OCR artifacts, misspellings) with
    valid ones. Two cases are suspicious:
    - Replacing a valid English word with a different word (hallucination)
    - Replacing a valid English word with an invalid word (introducing errors)
    """
    original_words = original.lower().split()
    cleaned_words = cleaned.lower().split()

    opcodes = difflib.SequenceMatcher(None, original_words, cleaned_words).get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'replace' or (i2 - i1) != (j2 - j1):
            continue
        for orig_word, new_word in zip(original_words[i1:i2], cleaned_words[j1:j2]):
            orig_clean = orig_word.strip('.,;:!?"\'()-[]')
            new_clean = new_word.strip('.,;:!?"\'()-[]')
            if orig_clean == new_clean:
                continue
            if word_validator.is_valid_word(orig_clean):
                # Original was a clean valid word — any substitution is suspicious
                return True
            if not word_validator.is_valid_word(new_clean):
                # Original was an OCR artifact but replacement is also invalid
                return True
    return False


class TextCleaner:
    """Cleans and classifies paragraph text using a local LLM via Ollama.

    Attributes:
        _model: The Ollama model to use for cleaning.
        _max_retries: Maximum number of retries if the response is malformed.
    """

    def __init__(self, model: str = 'llama3.1:8b', max_retries: int = 3,
                 temperature: float | None = None,
                 verbose: bool = False) -> None:
        """Initialise TextCleaner.

        Args:
            model: The Ollama model to use. Defaults to 'llama3.1:8b'.
            max_retries: Maximum number of retries on malformed responses. Defaults to 3.
            temperature: Sampling temperature passed to Ollama. Set to 0 for
                deterministic output (useful in tests). Defaults to None,
                which uses Ollama's built-in default.
            verbose: If True, prints LLM responses for debugging. Defaults to False.
        """
        self._model: str = model
        self._max_retries: int = max_retries
        self._temperature: float | None = temperature
        self._verbose: bool = verbose

    def clean(self, paragraph: str, page_context: str = "") -> tuple[str, ClassificationType]:
        """Clean and classify a paragraph of text.

        Args:
            paragraph: The paragraph text to clean and classify.
            page_context: The full text of the page for context. Defaults to empty string.

        Returns:
            A tuple of (cleaned_text, classification) where classification
            is 'body', 'footnote', or 'drop'.

        Raises:
            ValueError: If the LLM returns a malformed response after all retries.
        """
        if not paragraph.strip():
            return paragraph, 'drop'

        if page_context:
            user_content = f"Page context:\n{page_context}\n\nParagraph to clean and classify:\n{paragraph}"
        else:
            user_content = f"Paragraph to clean and classify:\n{paragraph}"

        for attempt in range(self._max_retries):
            cleaned_candidate: str | None = None
            try:
                options: dict[str, float] = {}
                if self._temperature is not None:
                    options['temperature'] = self._temperature
                response = ollama.chat(
                    model=self._model,
                    options=options or None,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_content}
                    ]
                )
                content: str = response['message']['content'].strip()
                vprint(self._verbose, f"LLM response: {repr(content)}")
                parsed = json.loads(content)

                cleaned_candidate = parsed['cleaned']
                classification: ClassificationType = parsed['classification']

                if classification not in ('body', 'footnote', 'drop'):
                    coerced = _coerce_classification(classification)
                    if coerced is None:
                        raise ValueError(f"Invalid classification: '{classification}'")
                    classification = coerced

                if paragraph and abs(len(cleaned_candidate) - len(paragraph)) / len(paragraph) > 0.10:
                    raise ValueError(f"Cleaned text size differs by more than 10% "
                                     f"(original={len(paragraph)}, cleaned={len(cleaned_candidate)})")

                if _has_suspicious_substitutions(paragraph, cleaned_candidate):
                    raise ValueError("LLM replaced valid English words — possible hallucination")

                cleaned_candidate = ' '.join(cleaned_candidate.split('\n'))
                return cleaned_candidate, classification

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                vprint(self._verbose, f"  → attempt {attempt + 1} rejected: {e}")
                if cleaned_candidate is not None:
                    vprint(self._verbose, f"  original: {paragraph}")
                    vprint(self._verbose, f"  cleaned:  {cleaned_candidate}")
                continue

        return paragraph, 'body'
