"""LLM-based text cleaner and classifier for book-to-audio conversion."""

from __future__ import annotations

import difflib
import json
import re
import unicodedata
from typing import Literal, TypeAlias

import ollama

from utils.general_utils import normalize_quotes
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
- Do NOT remove numbered list prefixes such as (1), (2), (3) from the start of a paragraph
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


def _normalize_dashes(word: str) -> str:
    """Normalize em-dashes and en-dashes to hyphens for word comparison.

    Prevents false positives when the LLM correctly converts a hyphen to an
    em-dash within a compound token (e.g. "false-as" → "false—as").
    """
    return word.replace('—', '-').replace('–', '-')


def _normalize_token(word: str) -> str:
    """Normalize a token for suspicious-substitution comparison.

    Removes punctuation from anywhere in the token (not just the ends),
    collapses all dash/hyphen variants to nothing, and ASCII-folds diacritics
    so that equivalent forms compare as equal:
      - "well-known" and "wellknown" → "wellknown"
      - "Eötvös" and "Eotvos" → "eotvos"
      - "star:✶" and "star:*" → "star" and "star" (after symbol stripping)

    Used only for the invalid→invalid check, not for word validation.
    """
    word = re.sub(r'[.,;:!?"\'()\[\]]', '', word)
    word = word.replace('—', '').replace('–', '').replace('-', '')
    word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('ascii')
    return word.lower()


_LIST_PREFIX_RE: re.Pattern[str] = re.compile(r'^(\(\d+\)|\d+[.)]) ')


def _restore_list_prefix(original: str, cleaned: str) -> str:
    """Restore a leading list prefix if the LLM dropped it.

    Paragraphs that begin with a numbered list marker such as "(3)" or "3."
    should have that prefix preserved. The LLM sometimes strips it, treating
    it as a stray marker. This function detects the pattern in the original
    and prepends it to the cleaned text when absent.

    Args:
        original: The original paragraph text.
        cleaned: The LLM-cleaned paragraph text.

    Returns:
        The cleaned text with the list prefix restored if it was dropped.
    """
    match = _LIST_PREFIX_RE.match(original)
    if match and not _LIST_PREFIX_RE.match(cleaned):
        return match.group(0) + cleaned
    return cleaned


def _restore_valid_words(original: str, cleaned: str, verbose: bool = False) -> str:
    """Restore any valid original words that the LLM substituted.

    When the LLM performs a 1:1 word substitution and the original word is
    valid English, the original word is silently restored while all other
    LLM changes (joined OCR splits, punctuation fixes, etc.) are kept.
    Hyphens and em-dashes are treated as equivalent during comparison.

    Args:
        original: The original paragraph text.
        cleaned: The LLM-cleaned paragraph text.
        verbose: If True, prints each word restoration to stdout.

    Returns:
        The cleaned text with any valid-word substitutions undone.
    """
    original_split = original.split()
    cleaned_split = cleaned.split()
    original_lower = [w.lower() for w in original_split]
    cleaned_lower = [w.lower() for w in cleaned_split]

    opcodes = difflib.SequenceMatcher(None, original_lower, cleaned_lower).get_opcodes()
    result: list[str] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'replace':
            result.extend(cleaned_split[j1:j2])
            continue

        orig_count = i2 - i1
        new_count = j2 - j1

        if orig_count == new_count:
            # 1:1 replacements — restore valid original words
            for k in range(orig_count):
                # Normalize quotes before stripping so that ASCII and smart/curly
                # quote variants (e.g. ' vs ‘) compare as equal. The LLM often
                # converts straight quotes to typographic quotes; without this step,
                # tokens like "'All" vs "‘All" would be seen as different words.
                orig_stripped = _normalize_dashes(
                    normalize_quotes(original_lower[i1 + k]).strip('.,;:!?"\'()-[]'))
                new_stripped = _normalize_dashes(
                    normalize_quotes(cleaned_lower[j1 + k]).strip('.,;:!?"\'()-[]'))
                orig_tok = original_lower[i1 + k]
                new_tok = cleaned_lower[j1 + k]
                if orig_stripped != new_stripped and word_validator.is_valid_word(orig_stripped):
                    # valid→something: restore the original word
                    vprint(verbose,
                           f"  → restored '{original_split[i1 + k]}' "
                           f"(LLM tried '{cleaned_split[j1 + k]}')")
                    result.append(original_split[i1 + k])
                elif (orig_stripped == new_stripped
                      and ('—' in orig_tok or '–' in orig_tok)
                      and '—' not in new_tok and '–' not in new_tok):
                    # em/en-dash downgraded to plain hyphen — restore to preserve
                    # typographic quality (upgrade direction is intentionally kept)
                    vprint(verbose,
                           f"  → restored '{original_split[i1 + k]}' "
                           f"(LLM tried '{cleaned_split[j1 + k]}')")
                    result.append(original_split[i1 + k])
                else:
                    result.append(cleaned_split[j1 + k])
        elif orig_count > new_count == 1:
            # N→1 merge — keep if the merged result is a valid word, a number, the
            # cleaned token is simply the original tokens concatenated (e.g. "i. e.," →
            # "i.e.,"), or the cleaned token has internal periods (abbreviation signal,
            # e.g. "Ph. D" → "Ph.D." or "U. S. A" → "U.S.A.").
            # Also keep em-dash upgrades ("criticism - and" → "criticism—and") and
            # hyphen compounding ("proof reading" → "proof-reading").
            merged = _normalize_dashes(cleaned_lower[j1].strip('.,;:!?"\'()-[]'))
            joined_orig = ''.join(original_split[i1:i2])
            cleaned_inner = cleaned_split[j1].strip('.,;:!?"\'()-[]')
            has_internal_period = '.' in cleaned_inner
            # Em-dash upgrade: dash-normalized cleaned token equals joined originals,
            # and the first original token starts with a letter (guards against a leading
            # standalone dash being glued to the next word, e.g. "- including" → "—including").
            is_dash_upgrade = (_normalize_dashes(cleaned_split[j1]) == joined_orig
                               and original_split[i1][0].isalpha())
            # Hyphen compounding: LLM joined two words with a hyphen ("proof reading" →
            # "proof-reading"). The hyphen-joined originals exactly match the cleaned token.
            is_hyphen_compound = '-'.join(original_split[i1:i2]) == cleaned_split[j1]
            if (word_validator.is_valid_word(merged)
                    or any(c.isdigit() for c in merged)
                    or cleaned_split[j1] == joined_orig
                    or has_internal_period
                    or is_dash_upgrade
                    or is_hyphen_compound):
                result.append(cleaned_split[j1])
            else:
                vprint(verbose,
                       f"  → restored {original_split[i1:i2]} "
                       f"(LLM tried '{cleaned_split[j1]}')")
                result.extend(original_split[i1:i2])
        else:
            # Other mismatches (1→N splits, N→M) — keep LLM version
            result.extend(cleaned_split[j1:j2])

    return ' '.join(result)


def _has_suspicious_substitutions(original: str, cleaned: str) -> bool:
    """Return True if the LLM made a substitution that looks like hallucination.

    We trust the LLM on all substitution types:
    - valid→valid: _restore_valid_words already silently reverts these
    - invalid→valid: a legitimate OCR fix — keep it
    - valid→invalid: _restore_valid_words already restores the original
    - invalid→invalid: we trust the LLM to do its best with OCR artifacts

    This function is retained as a hook for future checks but currently always
    returns False.
    """
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
                # vprint(self._verbose, f"LLM response: {repr(content)}")
                parsed = json.loads(content)

                cleaned_candidate = parsed['cleaned']
                classification: ClassificationType = parsed['classification']

                if classification not in ('body', 'footnote', 'drop'):
                    coerced = _coerce_classification(classification)
                    if coerced is None:
                        raise ValueError(f"Invalid classification: '{classification}'")
                    classification = coerced

                if classification != 'drop':
                    printable_len: int = sum(1 for c in paragraph if c.isprintable())
                    if printable_len and abs(len(cleaned_candidate) - printable_len) / printable_len > 0.10:
                        raise ValueError(f"Cleaned text size differs by more than 10% "
                                         f"(original printable={printable_len}, cleaned={len(cleaned_candidate)})")

                cleaned_candidate = _restore_valid_words(
                    paragraph, cleaned_candidate, verbose=self._verbose
                )
                cleaned_candidate = _restore_list_prefix(paragraph, cleaned_candidate)

                if _has_suspicious_substitutions(paragraph, cleaned_candidate):
                    raise ValueError("LLM made invalid→invalid substitution — possible hallucination")

                cleaned_candidate = ' '.join(cleaned_candidate.split('\n'))
                return cleaned_candidate, classification

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                vprint(self._verbose, f"  → attempt {attempt + 1} rejected: {e}")
                if cleaned_candidate is not None:
                    vprint(self._verbose, f"  original: {paragraph}")
                    vprint(self._verbose, f"  cleaned:  {cleaned_candidate}")
                continue

        return paragraph, 'body'
