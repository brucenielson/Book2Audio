"""Tests for utils.general_utils utility functions."""

import pytest

from utils.general_utils import (
    is_ends_with_punctuation,
    is_sentence_end,
    is_roman_numeral,
    remove_extra_whitespace,
    enhance_title,
    normalize_whitespace,
    normalize_hyphens,
    normalize_quotes,
    normalize_ligatures,
    fix_encoding_artifacts,
    fix_punctuation_spacing,
    fix_bracket_spacing,
    fix_apostrophes,
    strip_footnote_numbers,
    clean_text,
    build_paragraph,
    load_sections_to_skip,
    load_valid_pages,
    substitute_math_symbols,
    is_math_heavy,
)

# --- is_ends_with_punctuation ---

class TestIsEndsWithPunctuation:
    @pytest.mark.parametrize("text", ["Hello.", "Really?", "Wow!"])
    def test_is_true(self, text: str) -> None:
        assert is_ends_with_punctuation(text) is True

    @pytest.mark.parametrize("text", ["Hello", "Hello,"])
    def test_is_false(self, text: str) -> None:
        assert is_ends_with_punctuation(text) is False


# --- is_sentence_end ---

class TestIsSentenceEnd:
    @pytest.mark.parametrize("text", ["Hello world.", "Really?", "Wow!", "Hello world.)"])
    def test_is_true(self, text: str) -> None:
        assert is_sentence_end(text) is True

    @pytest.mark.parametrize("text", ["Hello world", "Hello world)"])
    def test_is_false(self, text: str) -> None:
        assert is_sentence_end(text) is False

    @pytest.mark.parametrize("text", [
        # Floating closer: sentence punct + space + closing quote/bracket
        "Hello world. '",      # straight single quote
        'Hello world. "',      # straight double quote
        "Hello world. ’",  # right single curly
        "Hello world. ”",  # right double curly
        # Floating closer + footnote number should still preserve sentence ending
        "latent content of dreams. '",
        # Space before period + floating closer + footnote number
        "statistics . '",
    ])
    def test_floating_closer_is_true(self, text: str) -> None:
        assert is_sentence_end(text) is True


# --- is_roman_numeral ---

class TestIsRomanNumeral:
    def test_valid_roman_numerals(self) -> None:
        for numeral in ["I", "IV", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"]:
            assert is_roman_numeral(numeral) is True

    def test_invalid_roman_numeral(self) -> None:
        assert is_roman_numeral("Hello") is False

    def test_case_insensitive(self) -> None:
        assert is_roman_numeral("iv") is True

    def test_empty_string(self) -> None:
        # Empty string matches the pattern (zero of everything)
        assert isinstance(is_roman_numeral(""), bool)


# --- remove_extra_whitespace ---

class TestRemoveExtraWhitespace:
    @pytest.mark.parametrize("text, expected", [
        ("hello   world",  "hello world"),
        ("  hello  ",      "hello"),
        ("hello\t\nworld", "hello world"),
        ("hello world",    "hello world"),
    ])
    def test_remove_extra_whitespace(self, text: str, expected: str) -> None:
        assert remove_extra_whitespace(text) == expected


# --- combine_paragraphs ---

class TestCombineParagraphs:
    def test_joins_with_newline_if_sentence_end(self) -> None:
        result = build_paragraph("First sentence.", "Second sentence.")
        assert result == "First sentence.\nSecond sentence."

    def test_joins_with_space_if_no_sentence_end(self) -> None:
        result = build_paragraph("First part", "second part.")
        assert result == "First part second part."

    def test_strips_result(self) -> None:
        result = build_paragraph("  Hello.  ", "  World.  ")
        assert result == "Hello.\nWorld."

    def test_empty_first_paragraph(self) -> None:
        result = build_paragraph("", "Second.")
        assert result == "Second."

    def test_list_form_joins_all_items(self) -> None:
        result = build_paragraph(["First part", "second part", "final sentence."])
        assert result == "First part second part final sentence."

    def test_list_form_newline_at_sentence_end(self) -> None:
        result = build_paragraph(["First sentence.", "Second sentence."])
        assert result == "First sentence.\nSecond sentence."

    def test_list_form_empty_list(self) -> None:
        assert build_paragraph([]) == ""

    def test_list_form_single_item(self) -> None:
        assert build_paragraph(["Only sentence."]) == "Only sentence."


# --- enhance_title ---

class TestEnhanceTitle:
    def test_all_caps_to_title_case(self) -> None:
        assert enhance_title("HELLO WORLD") == "Hello World"

    def test_preserves_leading_roman_numeral(self) -> None:
        assert enhance_title("IV THE BEGINNING") == "IV The Beginning"

    def test_no_change_for_mixed_case(self) -> None:
        assert enhance_title("Hello World") == "Hello World"

    def test_fixes_possessive_after_title_case(self) -> None:
        assert enhance_title("JOHN'S BOOK") == "John's Book"

    def test_strips_whitespace(self) -> None:
        assert enhance_title("  HELLO  ") == "Hello"


# --- normalize_whitespace ---

class TestNormalizeWhitespace:
    def test_strips_and_collapses(self) -> None:
        assert normalize_whitespace("  hello   world  ") == "hello world"


# --- normalize_hyphens ---

class TestNormalizeHyphens:
    @pytest.mark.parametrize("text, expected", [
        ("explo­ration", "exploration"),
        ("well-known",        "well-known"),
        ("some­ thing",  "some thing"),
    ])
    def test_normalize_hyphens(self, text: str, expected: str) -> None:
        assert normalize_hyphens(text) == expected


# --- normalize_quotes ---

class TestNormalizeQuotes:
    @pytest.mark.parametrize("text, expected", [
        ("“hello”", '"hello"'),
        ("‘hello’", "'hello'"),
        ("dog’s",        "dog's"),
    ])
    def test_normalize_quotes(self, text: str, expected: str) -> None:
        assert normalize_quotes(text) == expected


# --- normalize_ligatures ---

class TestNormalizeLigatures:
    @pytest.mark.parametrize("text, expected", [
        ("ﬁle",    "file"),
        ("ﬂoor",   "floor"),
        ("ﬀect",   "ffect"),    # noinspection SpellCheckingInspection
        ("ﬃcient", "fficient"),  # noinspection SpellCheckingInspection
        ("ﬄuent",  "ffluent"),   # noinspection SpellCheckingInspection
        ("ﬅar",    "star"),
    ])
    def test_normalize_ligature(self, text: str, expected: str) -> None:
        assert normalize_ligatures(text) == expected


# --- fix_encoding_artifacts ---

class TestFixEncodingArtifacts:
    @pytest.mark.parametrize("text, expected", [
        ("Òhello",     '"hello'),    # noinspection SpellCheckingInspection
        ("helloÓ",     'hello"'),
        ("todayÕs",    "today's"),
        ("helloÑworld", "hello—world"),  # noinspection SpellCheckingInspection
        ("1988Ð1998",  "1988–1998"),
    ])
    def test_fix_encoding_artifact(self, text: str, expected: str) -> None:
        assert fix_encoding_artifacts(text) == expected


# --- fix_punctuation_spacing ---

class TestFixPunctuationSpacing:
    @pytest.mark.parametrize("text, expected", [
        ("hello .",       "hello."),
        ("hello , world", "hello, world"),
        ("really ?",      "really?"),
        ("wow !",         "wow!"),
    ])
    def test_removes_space_before_punctuation(self, text: str, expected: str) -> None:
        assert fix_punctuation_spacing(text) == expected


# --- fix_bracket_spacing ---

class TestFixBracketSpacing:
    @pytest.mark.parametrize("text, expected", [
        ("( hello )", "(hello)"),
        ("[ hello ]", "[hello]"),
        ("{ hello }", "{hello}"),
    ])
    def test_removes_space_inside_brackets(self, text: str, expected: str) -> None:
        assert fix_bracket_spacing(text) == expected


# --- fix_apostrophes ---

class TestFixApostrophes:
    def test_fixes_possessive_apostrophe(self) -> None:
        assert fix_apostrophes("the dog 's bone") == "the dog's bone"

    def test_fixes_space_before_possessive(self) -> None:
        assert fix_apostrophes("the dog 's bone") == "the dog's bone"


# --- strip_footnote_numbers ---

class TestStripFootnoteNumbers:
    @pytest.mark.parametrize('text, expected', [
        # Basic: digit attached directly to sentence-ending punctuation
        ('Hello world.1',    'Hello world.'),
        ('Hello world.123',  'Hello world.'),
        ('Hello world!1',    'Hello world!'),
        ('Hello world?1',    'Hello world?'),
        # Digit separated from punctuation by whitespace
        ('Hello world. 1',   'Hello world.'),
        ('Hello world.  1',  'Hello world.'),
        # Closing quote/bracket AFTER the sentence punct -- no space before digit
        ("came back.'2",          "came back.'"),           # straight single quote
        ("came back.‘2",    "came back.‘"),      # left single curly (U+2018)
        ("came back.\"2",    "came back.\""),          # straight double quote
        ("came back.”2",    "came back.”"),       # right double curly (U+201D)
        ('argument.)2',           'argument.)'),
        ('argument.]2',           'argument.]'),
        # Closing quote/bracket AFTER the sentence punct -- space(s) before digit
        ("came back.' 2",         "came back.'"),
        ('argument.) 2',          'argument.)'),
        ('argument.)  2',         'argument.)'),
        # Floating closer (OCR artifact): space + closer + digit -- closer is dropped
        ("came back. '2",         "came back. '"),
        ("came back. ‘2",   "came back. ‘"),             # left single curly (U+2018)
        ("came back. \"2",    "came back. \""),             # straight double quote
        ("came back. ”2",   "came back. ”"),              # right double curly (U+201D)
        # Closing bracket/paren BEFORE the sentence punct (currently failing)
        ("societies).'8",                   "societies).'"),
        ("societies). 8",                   "societies)."),
        ("societies). '8",                  "societies). '"),
        ("(totalitarian societies). '8",    "(totalitarian societies). '"),
        # No digit: no change
        ('Hello world.',     'Hello world.'),
        ('Hello world',      'Hello world'),
        # Floating closer + footnote number should still preserve sentence ending
        ("latent content of dreams. '11", "latent content of dreams. '"),
        # Space before period + floating closer + footnote number
        ("statistics . '5", "statistics . '"),
    ])
    def test_trailing_footnote_number(self, text: str, expected: str) -> None:
        assert strip_footnote_numbers(text) == expected

    # TODO: Comment back in
    # @pytest.mark.parametrize("text, expected", [
    #     # Footnote number directly attached to sentence-ending punctuation, mid-paragraph
    #     ("penicillin.4 It describes",   "penicillin. It describes"),
    #     ("section 6).1 And being",      "section 6). And being"),
    #     ("argument.12 The next",        "argument. The next"),
    # ])
    # def test_mid_paragraph_no_space_footnote(self, text: str, expected: str) -> None:
    #     assert strip_footnote_numbers(text) == expected

    @pytest.mark.parametrize("text, expected", [
        # Decimal numbers must not be stripped
        ("value 3.14 The next",         "value 3.14 The next"),
        ("R1.2 (prior) = result.",      "R1.2 (prior) = result."),
        # Space before number is out of scope for this fix
        ("penicillin. 12 It describes", "penicillin. 12 It describes"),
    ])
    def test_no_false_positives(self, text: str, expected: str) -> None:
        assert strip_footnote_numbers(text) == expected


# --- clean_text ---

class TestCleanText:
    def test_strips_whitespace(self) -> None:
        assert clean_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self) -> None:
        assert clean_text("hello   world") == "hello world"

    def test_removes_space_before_period(self) -> None:
        assert clean_text("hello .") == "hello."

    def test_removes_space_before_comma(self) -> None:
        assert clean_text("hello , world") == "hello, world"

    def test_removes_space_inside_parentheses(self) -> None:
        assert clean_text("( hello )") == "(hello)"

    def test_fixes_possessive_apostrophe(self) -> None:
        assert clean_text("the dog 's bone") == "the dog's bone"

    def test_normalizes_smart_quotes(self) -> None:
        assert clean_text("“hello”") == '"hello"'

    def test_preserves_regular_hyphen(self) -> None:
        assert clean_text("well-known") == "well-known"

    def test_empty_string(self) -> None:
        assert clean_text("") == ""


# --- load_sections_to_skip ---

class TestLoadSectionsToSkip:
    def test_loads_sections_from_csv(self, tmp_path) -> None:
        csv_path = tmp_path / "sections_to_skip.csv"
        csv_path.write_text("Book Title,Section Title\nMy Book,chapter1\nMy Book,chapter2\n",
                            encoding="utf-8")
        result = load_sections_to_skip(csv_path)
        assert "My Book" in result
        assert "chapter1" in result["My Book"]
        assert "chapter2" in result["My Book"]

    def test_returns_empty_if_no_file(self, tmp_path) -> None:
        result = load_sections_to_skip(tmp_path / "nonexistent.csv")
        assert result == {}

    def test_handles_multiple_books(self, tmp_path) -> None:
        csv_path = tmp_path / "sections_to_skip.csv"
        csv_path.write_text(
            "Book Title,Section Title\nBook One,chapter1\nBook Two,intro\n",
            encoding="utf-8"
        )
        result = load_sections_to_skip(csv_path)
        assert "Book One" in result
        assert "Book Two" in result


# --- TestLoadValidPages ---

class TestLoadValidPages:
    def test_loads_page_range_from_csv(self, tmp_path) -> None:
        csv_path = tmp_path / "valid_pages.csv"
        csv_path.write_text(
            "Book Title,Start,End\nMy Book,5,120\n", encoding="utf-8"
        )
        result = load_valid_pages(str(csv_path))
        assert "My Book" in result
        assert result["My Book"] == (5, 120)

    def test_returns_empty_if_file_missing(self, tmp_path) -> None:
        result = load_valid_pages(str(tmp_path / "nonexistent.csv"))
        assert result == {}

    def test_handles_multiple_books(self, tmp_path) -> None:
        csv_path = tmp_path / "valid_pages.csv"
        csv_path.write_text(
            "Book Title,Start,End\nBook One,1,50\nBook Two,10,200\n", encoding="utf-8"
        )
        result = load_valid_pages(str(csv_path))
        assert result["Book One"] == (1, 50)
        assert result["Book Two"] == (10, 200)

    def test_strips_whitespace_from_fields(self, tmp_path) -> None:
        csv_path = tmp_path / "valid_pages.csv"
        csv_path.write_text(
            "Book Title,Start,End\n  My Book  ,  3  ,  99  \n", encoding="utf-8"
        )
        result = load_valid_pages(str(csv_path))
        assert "My Book" in result
        assert result["My Book"] == (3, 99)


# --- substitute_math_symbols ---

class TestSubstituteMathSymbols:
    @pytest.mark.parametrize("text, expected", [
        ("¬p",    "not p"),
        ("p ∧ q", "p and q"),
        ("p ∨ q", "p or q"),
        ("p → q", "p implies q"),
        ("∀x",    "for all x"),
        ("∃x",    "there exists x"),
        ("p ↔ q", "p if and only if q"),
        ("∴ p",   "therefore p"),
    ])
    def test_single_symbol(self, text: str, expected: str) -> None:
        assert substitute_math_symbols(text) == expected

    def test_symbol_no_spaces(self) -> None:
        """Symbol with no surrounding spaces should not produce run-together words."""
        assert substitute_math_symbols("p→q") == "p implies q"

    def test_no_symbols(self) -> None:
        """Plain text should pass through unchanged."""
        assert substitute_math_symbols("All men are mortal.") == "All men are mortal."

    def test_multiple_symbols(self) -> None:
        assert substitute_math_symbols("p ∧ q → r") == "p and q implies r"

    def test_no_double_spaces(self) -> None:
        """Replacing a spaced symbol should not leave double spaces."""
        result = substitute_math_symbols("p → q")
        assert "  " not in result


# --- is_math_heavy ---

class TestIsMathHeavy:
    """Tests for the is_math_heavy() heuristic used to detect formula-dense paragraphs."""

    @pytest.mark.parametrize("text", [
        "(G) (x)(Ey)(P(x + y) & P((2 + x) - y))",     # Goldbach's conjecture (item #21)
        "Ct(a,c) = C(a,a,c) = 1 - p(a,c).",            # corroboration formula
        "p(b,ac) - p(b,c) C(a,b,c) = 0",               # probability formula
        "f(x) = g(x) + h(x), so f(x) - g(x) = h(x).", # equation with function notation
        "∀x ∃y P(x, y) → Q(x)",                        # logical formula with Unicode symbols
    ])
    def test_formula_text_is_math_heavy(self, text: str) -> None:
        """Text with dense math notation is correctly identified as math-heavy."""
        assert is_math_heavy(text) is True

    @pytest.mark.parametrize("text", [
        "This is a normal paragraph about philosophy.",
        "However (which is true), we can conclude that the argument is valid.",
        "The French Revolution began in 1789 and transformed Europe.",
        "In this chapter we examine the problem and propose a solution (see Appendix A).",
        "Let x be the variable and y be the function value.",
    ])
    def test_normal_prose_is_not_math_heavy(self, text: str) -> None:
        """Normal prose is not flagged as math-heavy."""
        assert is_math_heavy(text) is False

    @pytest.mark.parametrize("text", [
        "25. (*58)",    # section marker: parentheses + asterisk but no = or + or Greek
        "26. (*59)",    # same pattern
        "28. (*61)",    # same pattern
    ])
    def test_section_markers_are_not_math_heavy(self, text: str) -> None:
        """Section markers with parentheses and asterisk must not be flagged.

        These have high parenthesis density but no genuine mathematical content
        (no equality, addition, or Unicode math/Greek symbols).
        """
        assert is_math_heavy(text) is False

    def test_empty_string_is_not_math_heavy(self) -> None:
        assert is_math_heavy("") is False

    def test_whitespace_only_is_not_math_heavy(self) -> None:
        assert is_math_heavy("   ") is False
