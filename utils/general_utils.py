"""General utility functions for text processing and file I/O."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from utils.logging_utils import vprint
from word_validator import word_validator as _word_validator


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
_CLOSING: frozenset[str] = frozenset(')}]\'"’”')


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
    # Walk backward past any trailing closing brackets/quotes and spaces,
    # interleaved. OCR often inserts spaces between nested closers or after
    # the final closer (e.g. 'dreams. " )' or 'replaced. ').
    i = len(text) - 1
    while i >= 0 and (text[i] in _CLOSING or text[i] == ' '):
        i -= 1
    return i >= 0 and text[i] in _SENTENCE_END


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
    # Remove trailing footnote number after sentence-ending punctuation,
    # optionally followed by closing quotes or brackets before the number.
    # e.g. "Hello world. 1" -> "Hello world."
    #      "hit you.'2"     -> "hit you.'"
    #      "argument.) 2"   -> "argument.)"
    #      "defined'. 3)"   -> "defined'.)"
    _cl = chr(0x27) + '"' + chr(0x2019) + chr(0x201D) + r')\]'
    _apos = chr(0x27) + chr(0x2019)
    # Case A: number at bare end of string (no closer after digit). No lookbehind needed —
    # a bare trailing number is unambiguously a footnote regardless of word length.
    # e.g. "hypothesis h. 1" -> "hypothesis h."
    p_str = re.sub(rf'(\w[{_cl}]*\s*[.!?][{_cl}]*)\s*\d{{1,2}}\s*$',
                   r'\1', p_str)
    # Case B: number followed by a closer (bracket/quote). Require the word before
    # sentence-ending punctuation to be multi-char or follow an apostrophe, to avoid
    # stripping page numbers in parenthetical citations like "(Harris 2006, p. 25)".
    p_str = re.sub(rf'((?<=[\w{_apos}])\w[{_cl}]*\s*[.!?][{_cl}]*)\s*\d{{1,2}}\s*([{_cl}]+)\s*$',
                   r'\1\2', p_str)
    # Strip trailing footnote when a space separates sentence-ending punctuation
    # from a closing quote before the number, e.g. 'hit you. "2' -> 'hit you.'
    # Same two-case split as above.
    p_str = re.sub(rf'(\w[{_cl}]*\s*[.!?]\s+[{_cl}]*)\s*\d{{1,2}}\s*$',
                   r'\1', p_str)
    p_str = re.sub(rf'((?<=[\w{_apos}])\w[{_cl}]*\s*[.!?]\s+[{_cl}]*)\s*\d{{1,2}}\s*([{_cl}]+)\s*$',
                   r'\1\2', p_str)
    # Remove footnote numbers directly attached (no space) to sentence-ending
    # punctuation, mid-paragraph or at end of string.
    # e.g. "section).1 And" -> "section). And", "penicillin.4 It" -> "penicillin. It"
    # The (?<!\d) guard on '.' prevents stripping from decimal numbers like 3.14.
    # Quotes only count as footnote separators when directly preceded by sentence-ending
    # punctuation (with or without an intervening space), preventing false positives on
    # opening quotes like "The '10 percent'". Both straight (' ") and right curly (' ")
    # closing quotes are included. The space variant handles OCR output like ". '2" where
    # fix_punctuation_spacing would later collapse the space but runs after this function.
    _q = chr(0x27) + '"' + chr(0x2019) + chr(0x201D)
    p_str = re.sub(rf"(?:(?<!\d)(?<![.][A-Z])\.|[!?)\]]|(?<=[.!?])[{_q}]|(?<=[.!?]\s)[{_q}])\d{{1,2}}(?=\s|$)",
                   lambda m: m.group(0)[0], p_str)
    # Handle multi-space OCR artifacts between sentence punctuation and closing quote,
    # e.g. "valid.  '9 Thus" -> "valid.  ' Thus". Python re lookbehinds are fixed-width
    # so two-or-more spaces cannot be expressed as a lookbehind — a capturing group is used.
    p_str = re.sub(rf"([.!?]\s{{2,}}[{_q}])\d{{1,2}}(?=\s|$)", r'\1', p_str)
    # Handle closing quote directly after sentence punct, then a space, then footnote number.
    # e.g. "'finest hour.' 19 Pamela" -> "'finest hour.' Pamela"
    # Seen in legal writing OCR where superscript refs are separated from closing quotes.
    # Limited to \d{1,2} so 4-digit years (1990, 2024) are never stripped.
    p_str = re.sub(rf"((?<=[.!?])[{_q}])\s+\d{{1,2}}(?=\s|$)", r'\1', p_str)
    # Handle any quote character directly after a word character followed by a footnote
    # number, e.g. "'the dice-playing god'2 is" -> "'the dice-playing god' is".
    # All six quote variants are included (straight, left curly, right curly, single and
    # double). Apostrophe false positives are not a concern: "can't2" matches "t'2" and
    # returns "t'", correctly giving "can't".
    _all_q = _q + chr(0x2018) + chr(0x201C)
    p_str = re.sub(rf"(\w[{_all_q}])\d{{1,2}}(?=\s|$)", r'\1', p_str)
    # Strip footnote refs of the form: word(4+ chars). SPACE number SPACE Uppercase/quote.
    # e.g. "Casey. 46 After" -> "Casey. After", "minorities. 81 'We" -> "minorities. 'We"
    # The 4-char lookbehind excludes common abbreviations: p., v., ch., vol., art., sec.
    # The uppercase/opening-quote lookahead excludes lowercase continuations like
    # "p. 12 for details" even when the preceding word is long enough to match.
    _open_q = chr(0x2018) + chr(0x201C) + "'\""
    p_str = re.sub(rf"(?<=\w\w\w\w)\.\s+\d{{1,2}}(?=\s[A-Z{_open_q}])",
                   '.', p_str)
    return p_str


def strip_latex_superscripts(p_str: str) -> str:
    """Remove LaTeX-style superscript footnote markers from text.

    Docling renders inline superscript references from PDFs as LaTeX tokens,
    e.g. $^{2} for footnote 2, or $^{1}$^{0} for footnote 10 (two consecutive
    single-digit tokens). These are rendering artifacts that should never appear
    in output text.

    # TODO: Consider moving this to DoclingParser._get_processed_texts, since
    #       $^{N} is a Docling-specific artifact rather than a general text issue.

    Args:
        p_str: The raw string to clean.

    Returns:
        The string with all $^{N} sequences removed.
    """
    return re.sub(r'(\$\^\{\d\})+', '', p_str)


def clean_text(p_str: str, remove_footnotes: bool = False) -> str:
    """Clean and normalize a text string.

    Applies a pipeline of normalization steps in order: LaTeX superscript
    removal, ligature normalization, encoding artifact correction, punctuation
    spacing, bracket spacing, apostrophe normalization, dash-hyphen detection,
    quote normalization, and whitespace normalization. Optionally strips
    trailing footnote numbers before the main pipeline runs.

    Args:
        p_str: The raw string to clean.
        remove_footnotes: If True, strips trailing footnote numbers before
                          other cleaning steps. Defaults to False.

    Returns:
        The cleaned and normalized string.
    """
    p_str = strip_latex_superscripts(p_str)
    if remove_footnotes:
        p_str = strip_footnote_numbers(p_str)
    p_str = normalize_ligatures(p_str)
    p_str = fix_encoding_artifacts(p_str)
    p_str = fix_punctuation_spacing(p_str)
    p_str = fix_bracket_spacing(p_str)
    p_str = fix_apostrophes(p_str)
    p_str = _word_validator.fix_dash_hyphens(p_str)
    p_str = normalize_quotes(p_str)
    p_str = normalize_whitespace(p_str)
    return p_str.strip()


# Minimum character count (including spaces) for a string to be considered a formula.
# Shorter strings like "(1)" or "(G')" are reference labels, not real expressions.
MIN_FORMULA_LENGTH: int = 5

# Characters that are rare in prose but common in mathematical notation.
# Parentheses and brackets are the dominant signal — formula-dense text like
# "(G) (x)(Ey)(P(x + y) & P((2 + x) - y))" has ~45% math chars; a sentence
# like "However (which is true), we conclude." has ~3%.
# Deliberately excludes '-' (too common as a hyphen) and ':' (too common in prose).
_MATH_CHARS: frozenset[str] = frozenset(
    '=+*/^&|<>()[]{}' +                            # operators and brackets
    '∀∃∧∨¬→←↔∈∉⊂⊃⊆⊇≡≤≥≠∑∏∫√∞±·×÷' +            # Unicode logic/math
    'αβγδεζηθικλμνξπρστυφχψω' +                   # Greek lowercase
    'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ'                     # Greek uppercase
)

# Subset of _MATH_CHARS that can only appear in genuine mathematical content.
# Density alone is insufficient — "25. (*58)" (a section marker) has high
# parenthesis density but no actual mathematical meaning. Requiring at least
# one strong signal prevents these false positives.
_STRONG_MATH_SIGNALS: frozenset[str] = frozenset(
    '=+<>' +                                        # equality, addition, inequalities
    '∀∃∧∨¬→←↔∈∉⊂⊃⊆⊇≡≤≥≠∑∏∫√∞±·×÷' +            # Unicode logic/math
    'αβγδεζηθικλμνξπρστυφχψω' +                   # Greek lowercase
    'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ'                     # Greek uppercase
)


def is_math_heavy(text: str, threshold: float = 0.20) -> bool:
    """Return True if text appears to be a math-heavy formula or expression.

    Two conditions must both hold:
    1. At least `threshold` fraction of non-space characters are in _MATH_CHARS
       (operators, brackets, Unicode logic/math symbols, Greek letters).
    2. At least one character from _STRONG_MATH_SIGNALS is present — equality,
       addition, Unicode logic/math, or Greek letters. This prevents false
       positives on section markers like "25. (*58)" which have many parentheses
       but no genuine mathematical content.

    Args:
        text: The paragraph text to test.
        threshold: Minimum fraction of non-space characters that must be
                   math-related. Defaults to 0.20 (20%).

    Returns:
        True if the paragraph is math-heavy, False otherwise.
    """
    if len(text) < MIN_FORMULA_LENGTH:
        return False
    non_space = [c for c in text if c != ' ']
    if not non_space:
        return False
    math_count = sum(1 for c in non_space if c in _MATH_CHARS)
    if math_count / len(non_space) < threshold:
        return False
    return any(c in _STRONG_MATH_SIGNALS for c in text)


def substitute_math_symbols(text: str) -> str:
    """Replace math/logical symbols with spoken English equivalents for TTS.

    Intended as a pre-processing step before audio generation. Each symbol is
    replaced with a word (padded with spaces), and extra whitespace is collapsed
    afterwards so symbols without surrounding spaces (e.g. 'p→q') still produce
    clean output ('p implies q').

    Args:
        text: The text to process.

    Returns:
        The text with math/logical symbols replaced by their spoken equivalents.
    """
    _SUBSTITUTIONS: list[tuple[str, str]] = [
        ('¬', ' not '),
        ('∧', ' and '),
        ('∨', ' or '),
        ('→', ' implies '),
        ('∀', ' for all '),
        ('∃', ' there exists '),
        ('↔', ' if and only if '),
        ('∴', ' therefore '),
        ('∵', ' because '),
        ('⊥', ' contradiction '),
        ('⊃', ' implies '),
        ('≡', ' is equivalent to '),
    ]
    for symbol, replacement in _SUBSTITUTIONS:
        text = text.replace(symbol, replacement)
    return remove_extra_whitespace(text)


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
