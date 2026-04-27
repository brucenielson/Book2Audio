"""General utility functions for text processing and file I/O."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from utils.logging_utils import vprint


def print_debug_results(results: dict[str, Any],
                        include_outputs_from: set[str] | None = None,
                        verbose: bool = True) -> None:
    """Print a filtered and hierarchical view of debug results.

    Args:
        results: The full results dict to display.
        include_outputs_from: If provided, only keys present in this set are printed.
        verbose: If False, nothing is printed. Defaults to True.
    """
    level: int = 1
    if verbose and include_outputs_from is not None:
        results_filtered = {k: v for k, v in results.items() if k in include_outputs_from}
        if results_filtered:
            vprint(verbose)
            vprint(verbose, "Debug Results:")
            _print_hierarchy(results_filtered, level, verbose)


def _print_hierarchy(data: dict[str, Any], level: int, verbose: bool = True) -> None:
    """Recursively print a nested dict structure with level indentation.

    Args:
        data: The dict to print.
        level: The current nesting level, used for labeling output lines.
        verbose: If False, nothing is printed. Defaults to True.
    """
    for key, value in data.items():
        if level == 1:
            vprint(verbose)
        vprint(verbose, f"Level {level}: {key}")
        if isinstance(value, dict):
            _print_hierarchy(value, level + 1, verbose)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                vprint(verbose, f"Level {level + 1}: Item {index + 1}")
                if isinstance(item, dict):
                    _print_hierarchy(item, level + 2, verbose)
                else:
                    vprint(verbose, item)
        else:
            vprint(verbose, value)


def load_valid_pages(skip_file: str) -> dict[str, tuple[int, int]]:
    """Load a CSV file mapping book titles to valid page ranges.

    The CSV must have columns 'Book Title', 'Start', and 'End'.

    Args:
        skip_file: Path to the CSV file.

    Returns:
        A dict mapping book titles to (start_page, end_page) tuples.
    """
    book_pages: dict[str, tuple[int, int]] = {}
    skip_file_path = Path(skip_file)
    if skip_file_path.exists():
        with open(skip_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader: csv.DictReader[str] = csv.DictReader(csvfile)
            row: dict[str, str]
            for row in reader:
                book_title: str = row['Book Title'].strip()
                start: str = row['Start'].strip()
                end: str = row['End'].strip()
                if book_title and start and end:
                    book_pages[book_title] = (int(start), int(end))
    return book_pages


def load_sections_to_skip(csv_path: Path, verbose: bool = False) -> dict[str, set[str]]:
    """Load a CSV file listing book sections to skip during parsing.

    The CSV file must have columns 'Book Title' and 'Section Title'.

    Args:
        csv_path: Path to the CSV file.
        verbose: If True, prints a summary of what was loaded. Defaults to False.

    Returns:
        A dict mapping book titles to sets of section IDs to skip.
    """
    sections_to_skip: dict[str, set[str]] = {}
    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader: csv.DictReader[str] = csv.DictReader(csvfile)
            row: dict[str, str]
            for row in reader:
                book_title: str = row['Book Title'].strip()
                section_title: str = row['Section Title'].strip()
                if book_title and section_title:
                    if book_title not in sections_to_skip:
                        sections_to_skip[book_title] = set()
                    sections_to_skip[book_title].add(section_title)
        skip_count: int = sum(len(sections) for sections in sections_to_skip.values())
        vprint(verbose, f"Loaded {skip_count} sections to skip.")
    else:
        vprint(verbose, "No sections_to_skip.csv file found. Processing all sections.")
    return sections_to_skip


def is_roman_numeral(s: str) -> bool:
    # noinspection SpellCheckingInspection
    """Check if a string is a Roman numeral.

    The check is case-insensitive and matches standard Roman numerals
    from I to MMMCMXCIX.

    Args:
        s: The string to check.

    Returns:
        True if the string is a valid Roman numeral, False otherwise.
    """
    roman_numeral_pattern: str = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))


def enhance_title(text: str) -> str:
    """Convert all-caps titles to title case, preserving leading Roman numerals.

    Args:
        text: The title text to enhance.

    Returns:
        The title with appropriate casing applied.
    """
    text = text.strip()
    if text.isupper() and not is_roman_numeral(text):
        first_word = text.split(' ', 1)[0]
        if is_roman_numeral(first_word) and first_word != text:
            text = first_word + text[len(first_word):].title()
        else:
            text = text.title()
        text = text.replace("'S", "'s")
        text = text.replace("\u2019S", "\u2019s")
    return text


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple consecutive whitespace characters into a single space.

    Also strips leading and trailing whitespace.

    Args:
        text: The string to process.

    Returns:
        The string with all whitespace runs collapsed to a single space.
    """
    return ' '.join(text.split())


def normalize_whitespace(p_str: str) -> str:
    """Strip and collapse whitespace in a string.

    Converts the input to a string, strips leading and trailing whitespace,
    ensures UTF-8 encoding, and collapses internal whitespace runs to single spaces.

    Args:
        p_str: The string to process.

    Returns:
        The normalized string.
    """
    p_str = str(p_str).strip()
    p_str = p_str.encode('utf-8').decode('utf-8')
    return remove_extra_whitespace(p_str)


def normalize_hyphens(p_str: str) -> str:
    # noinspection SpellCheckingInspection
    """Remove soft hyphens (SHY, U+00AD) from a string.

    Soft hyphens are invisible line-break hints inserted by typesetters.
    In OCR'd text they appear where a word was broken across lines and
    should be removed to restore the original word.

    Args:
        p_str: The string to process.

    Returns:
        The string with soft hyphens removed.
    """
    p_str = p_str.replace("\u00ad", "")
    return p_str


def normalize_quotes(p_str: str) -> str:
    """Replace curly/smart quotes with straight ASCII equivalents.

    Args:
        p_str: The string to process.

    Returns:
        The string with smart quotes replaced by straight quotes.
    """
    p_str = p_str.replace('\u201c', '"').replace('\u201d', '"')  # " "
    p_str = p_str.replace('\u2018', "'").replace('\u2019', "'")  # ' '
    return p_str


def normalize_ligatures(p_str: str) -> str:
    """Replace common OCR ligature characters with their letter equivalents.

    Args:
        p_str: The string to process.

    Returns:
        The string with ligatures replaced.
    """
    p_str = p_str.replace('ﬁ', 'fi')
    p_str = p_str.replace('ﬂ', 'fl')
    p_str = p_str.replace('ﬀ', 'ff')
    p_str = p_str.replace('ﬃ', 'ffi')
    p_str = p_str.replace('ﬄ', 'ffl')
    p_str = p_str.replace('ﬅ', 'st')
    return p_str


def fix_encoding_artifacts(p_str: str) -> str:
    """Replace common Mac Roman / Windows-1252 mojibake characters with correct equivalents.

    Args:
        p_str: The string to process.

    Returns:
        The string with encoding artifacts replaced.
    """
    p_str = p_str.replace('Ò', '"').replace('Ó', '"')  # curly double quotes
    p_str = p_str.replace('Õ', "'")                    # curly apostrophe
    p_str = p_str.replace('Ñ', '—')                    # em dash
    p_str = p_str.replace('Ð', '–')                    # en dash
    return p_str


def fix_punctuation_spacing(p_str: str) -> str:
    """Remove erroneous whitespace around punctuation marks.

    Handles spaces before periods, commas, question marks, exclamation points,
    and spaces between sentence-ending punctuation and closing quotes.

    Args:
        p_str: The string to process.

    Returns:
        The string with punctuation spacing corrected.
    """
    p_str = re.sub(r"([.!?]) '", r"\1'", p_str)  # Remove the space between punctuation (.!?) and '
    p_str = re.sub(r'([.!?]) "', r'\1"', p_str)  # Remove the space between punctuation (.!?) and "
    p_str = re.sub(r'\s+,', ',', p_str)  # Remove whitespace before a comma
    p_str = re.sub(r'(?<=\s)\.([a-zA-Z])', r'\1', p_str)  # Remove a period that follows a whitespace and comes before a letter
    p_str = re.sub(r'\s+\.', '.', p_str)  # Remove any whitespace before a period
    p_str = re.sub(r'\s+\?', '?', p_str)  # Remove any whitespace before a question mark
    p_str = re.sub(r'\s+!', '!', p_str)  # Remove any whitespace before an exclamation point
    return p_str


def fix_bracket_spacing(p_str: str) -> str:
    """Remove erroneous whitespace inside brackets and parentheses.

    Args:
        p_str: The string to process.

    Returns:
        The string with bracket spacing corrected.
    """
    p_str = re.sub(r'\s+\)', ')', p_str)  # Remove whitespace before a closing parenthesis
    p_str = re.sub(r'\s+]', ']', p_str)  # Remove whitespace before a closing square bracket
    p_str = re.sub(r'\s+}', '}', p_str)  # Remove whitespace before a closing curly brace
    p_str = re.sub(r'\(\s+', '(', p_str)  # Remove whitespace after an opening parenthesis
    p_str = re.sub(r'\[\s+', '[', p_str)  # Remove whitespace after an opening square bracket
    p_str = re.sub(r'\{\s+', '{', p_str)  # Remove whitespace after an opening curly brace
    return p_str


def fix_apostrophes(p_str: str) -> str:
    """Fix erroneous whitespace around possessive apostrophes.

    Handles two OCR artifacts: a space between an apostrophe and the following
    's', and a space before a possessive 's.

    Args:
        p_str: The string to process.

    Returns:
        The string with apostrophe spacing corrected.
    """
    p_str = re.sub(r"'\s+s(\s|[.,!?;:])", r"'s\1", p_str)
    p_str = re.sub(r"\s+'s(\s|$)", r"'s\1", p_str)
    return p_str


_SENTENCE_END: frozenset[str] = frozenset('.?!')
_CLOSING: frozenset[str] = frozenset({")", "}", "]", '"', "'", '”', '’'})


def is_ends_with_punctuation(text: str) -> bool:
    """Check if a string ends with sentence-ending punctuation.

    Args:
        text: The string to check.

    Returns:
        True if the string ends with a period, question mark, or exclamation point.
    """
    return bool(text) and text[-1] in _SENTENCE_END


def build_paragraph(paragraphs: list[str] | str, p2_str: str = "") -> str:
    """Build a single paragraph out of two strings.

    Accepts either a list of strings or two strings (legacy usage).
    If the first paragraph ends with sentence-ending punctuation, the two are
    joined with a newline. Otherwise, they are joined with a space, treating
    them as a continuation of the same sentence.

    Args:
        paragraphs: Either a list of strings to combine, or the first string
                    in a two-string combination.
        p2_str: The second string when called with two string arguments.

    Returns:
        The combined paragraph string, stripped of leading and trailing whitespace.
    """
    if isinstance(paragraphs, list):
        result: str = ""
        for p in paragraphs:
            result = build_paragraph(result, p)
        return result

    # Two-string usage
    p1_str = paragraphs.strip()
    p2_str = p2_str.strip()
    if not p1_str:
        return p2_str
    if is_sentence_end(p1_str):
        return (p1_str + "\n" + p2_str).strip()
    else:
        return (p1_str + " " + p2_str).strip()


def is_sentence_end(text: str) -> bool:
    """Check if a string ends with a complete sentence.

    Handles standard punctuation as well as closing brackets and quotes
    that follow sentence-ending punctuation.

    Args:
        text: The string to check.

    Returns:
        True if the string appears to end a complete sentence.
    """
    if not text:
        return False
    last = text[-1]
    if last in _SENTENCE_END:
        return True
    # Closing bracket/quote immediately after sentence-ending punctuation.
    return last in _CLOSING and len(text) >= 2 and text[-2] in _SENTENCE_END


def strip_footnote_numbers(p_str: str) -> str:
    """Remove footnote markers from paragraph text.

    Removes trailing footnote numbers that appear after sentence-ending
    punctuation, e.g. "minor religious sects. 1" -> "minor religious sects."
    Only strips numbers that are clearly footnote markers — i.e. a space
    followed by a number at the end of a sentence.

    Args:
        p_str: The paragraph string to clean.

    Returns:
        The string with footnote markers removed.
    """
    # Remove trailing footnote number after sentence-ending punctuation
    # e.g. "Hello world. 1" -> "Hello world."
    p_str = re.sub(r'(\w[.!?])\s*\d+\s*$', r'\1', p_str)
    return p_str


def clean_text(p_str: str, remove_footnotes: bool = False) -> str:
    """Clean and normalize a text string.

    Applies a pipeline of normalization steps in order: ligature normalization,
    encoding artifact correction, punctuation spacing, bracket spacing,
    apostrophe normalization, hyphen normalization, quote normalization,
    and whitespace normalization. Optionally strips trailing footnote numbers
    before the main pipeline runs.

    Args:
        p_str: The raw string to clean.
        remove_footnotes: If True, strips trailing footnote numbers before
                          other cleaning steps. Defaults to False.

    Returns:
        The cleaned and normalized string.
    """
    if remove_footnotes:
        p_str = strip_footnote_numbers(p_str)
    p_str = normalize_ligatures(p_str)
    p_str = fix_encoding_artifacts(p_str)
    p_str = fix_punctuation_spacing(p_str)
    p_str = fix_bracket_spacing(p_str)
    p_str = fix_apostrophes(p_str)
    p_str = normalize_hyphens(p_str)
    p_str = normalize_quotes(p_str)
    p_str = normalize_whitespace(p_str)
    return p_str.strip()


def extract_pdf_pages(source_path: str | Path,
                      dest_path: str | Path,
                      start_page: int,
                      end_page: int) -> Path:
    """Extract a range of pages from a PDF, preserving the text layer.

    Page numbers are physical (1-indexed), matching what Acrobat Reader
    shows in its page-count toolbar. Roman-numeral intro pages are still
    physical pages 1, 2, 3, … from the front of the file.

    Args:
        source_path: Path to the source PDF file.
        dest_path: Path to write the extracted PDF to.
        start_page: First physical page to include (1-indexed, inclusive).
        end_page: Last physical page to include (1-indexed, inclusive).

    Returns:
        The path of the written PDF.

    Raises:
        ValueError: If the page range is invalid for the document.
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    pdf = pdfium.PdfDocument(source_path)
    total = len(pdf)
    if start_page < 1 or end_page > total or start_page > end_page:
        raise ValueError(
            f"Invalid page range {start_page}–{end_page} for a {total}-page document."
        )
    indices = list(range(start_page - 1, end_page))  # convert to 0-indexed
    new_pdf = pdfium.PdfDocument.new()
    new_pdf.import_pages(pdf, indices)
    new_pdf.save(dest_path)
    print(f"Extracted pages {start_page}–{end_page} of {total} → {dest_path}")
    return dest_path
