import csv
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set, List


def print_debug_results(results: Dict[str, Any],
                        include_outputs_from: Optional[set[str]] = None,
                        verbose: bool = True) -> None:
    level: int = 1
    if verbose and include_outputs_from is not None:
        results_filtered = {k: v for k, v in results.items() if k in include_outputs_from}
        if results_filtered:
            print()
            print("Debug Results:")
            _print_hierarchy(results_filtered, level)


def _print_hierarchy(data: Dict[str, Any], level: int) -> None:
    for key, value in data.items():
        if level == 1:
            print()
        print(f"Level {level}: {key}")
        if isinstance(value, dict):
            _print_hierarchy(value, level + 1)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                print(f"Level {level + 1}: Item {index + 1}")
                if isinstance(item, dict):
                    _print_hierarchy(item, level + 2)
                else:
                    print(item)
        else:
            print(value)


def load_valid_pages(skip_file: str) -> Dict[str, Tuple[int, int]]:
    book_pages: Dict[str, Tuple[int, int]] = {}
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


def load_sections_to_skip(csv_path: Path) -> Dict[str, Set[str]]:
    """Load a CSV file listing book sections to skip during parsing.

    The CSV file must have columns 'Book Title' and 'Section Title'.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A dict mapping book titles to sets of section IDs to skip.
    """
    sections_to_skip: Dict[str, Set[str]] = {}
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
        print(f"Loaded {skip_count} sections to skip.")
    else:
        print("No sections_to_skip.csv file found. Processing all sections.")
    return sections_to_skip


def is_roman_numeral(s: str) -> bool:
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


def is_ends_with_punctuation(text: str) -> bool:
    """Check if a string ends with sentence-ending punctuation.

    Args:
        text: The string to check.

    Returns:
        True if the string ends with a period, question mark, or exclamation point.
    """
    return text.endswith(".") or text.endswith("?") or text.endswith("!")


def build_paragraph(paragraphs: List[str] | str, p2_str: str = "") -> str:
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
    has_end_punctuation: bool = is_ends_with_punctuation(text)
    # Does it end with a closing bracket, quote, etc.?
    ends_with_bracket: bool = (text.endswith(")")
                               or text.endswith("]")
                               or text.endswith("}")
                               or text.endswith("\"")
                               or text.endswith("\u201d")
                               or text.endswith("\u2019")
                               or text.endswith("\'"))
    return (has_end_punctuation or
            (ends_with_bracket and is_ends_with_punctuation(text[0:-1])))


def strip_footnote_numbers(p_str: str) -> str:
    """Remove trailing footnote reference numbers from a string.

    Strips digits from the end of the string until the string ends with
    valid sentence-ending punctuation or no more digits remain.

    Args:
        p_str: The string to process.

    Returns:
        The string with trailing footnote numbers removed.
    """
    while p_str and not is_sentence_end(p_str):
        last_char: str = p_str[-1]
        if not last_char.isdigit():
            break
        p_str = p_str[:-1].strip()
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
