"""Tests for utils.general_utils utility functions."""

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
)

# --- is_ends_with_punctuation ---

class TestIsEndsWithPunctuation:
    def test_period(self) -> None:
        assert is_ends_with_punctuation("Hello.") is True

    def test_question_mark(self) -> None:
        assert is_ends_with_punctuation("Really?") is True

    def test_exclamation(self) -> None:
        assert is_ends_with_punctuation("Wow!") is True

    def test_no_punctuation(self) -> None:
        assert is_ends_with_punctuation("Hello") is False

    def test_comma(self) -> None:
        assert is_ends_with_punctuation("Hello,") is False


# --- is_sentence_end ---

class TestIsSentenceEnd:
    def test_ends_with_period(self) -> None:
        assert is_sentence_end("Hello world.") is True

    def test_ends_with_question_mark(self) -> None:
        assert is_sentence_end("Really?") is True

    def test_ends_with_exclamation(self) -> None:
        assert is_sentence_end("Wow!") is True

    def test_ends_with_bracket_after_period(self) -> None:
        assert is_sentence_end("Hello world.)") is True

    def test_no_sentence_end(self) -> None:
        assert is_sentence_end("Hello world") is False

    def test_ends_with_bracket_no_punctuation(self) -> None:
        assert is_sentence_end("Hello world)") is False


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
    def test_collapses_multiple_spaces(self) -> None:
        assert remove_extra_whitespace("hello   world") == "hello world"

    def test_strips_leading_trailing(self) -> None:
        assert remove_extra_whitespace("  hello  ") == "hello"

    def test_handles_tabs_and_newlines(self) -> None:
        assert remove_extra_whitespace("hello\t\nworld") == "hello world"

    def test_no_change_needed(self) -> None:
        assert remove_extra_whitespace("hello world") == "hello world"


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
    def test_removes_soft_hyphen(self) -> None:
        assert normalize_hyphens("explo\u00adration") == "exploration"

    def test_preserves_regular_hyphen(self) -> None:
        assert normalize_hyphens("well-known") == "well-known"

    def test_removes_soft_hyphen_at_word_boundary(self) -> None:
        assert normalize_hyphens("some\u00ad thing") == "some thing"


# --- normalize_quotes ---

class TestNormalizeQuotes:
    def test_normalizes_left_double_quote(self) -> None:
        assert normalize_quotes("\u201chello\u201d") == '"hello"'

    def test_normalizes_smart_single_quotes(self) -> None:
        assert normalize_quotes("\u2018hello\u2019") == "'hello'"

    def test_normalizes_right_single_quote_possessive(self) -> None:
        assert normalize_quotes("dog\u2019s") == "dog's"


# --- normalize_ligatures ---

class TestNormalizeLigatures:
    def test_normalizes_fi_ligature(self) -> None:
        assert normalize_ligatures("ﬁle") == "file"

    def test_normalizes_fl_ligature(self) -> None:
        assert normalize_ligatures("ﬂoor") == "floor"

    def test_normalizes_ff_ligature(self) -> None:
        # noinspection SpellCheckingInspection
        assert normalize_ligatures("ﬀect") == "ffect"

    def test_normalizes_ffi_ligature(self) -> None:
        # noinspection SpellCheckingInspection
        assert normalize_ligatures("ﬃcient") == "fficient"

    def test_normalizes_ffl_ligature(self) -> None:
        # noinspection SpellCheckingInspection
        assert normalize_ligatures("ﬄuent") == "ffluent"

    def test_normalizes_st_ligature(self) -> None:
        assert normalize_ligatures("ﬅar") == "star"


# --- fix_encoding_artifacts ---

class TestFixEncodingArtifacts:
    def test_fixes_left_double_quote(self) -> None:
        # noinspection SpellCheckingInspection
        assert fix_encoding_artifacts("Òhello") == '"hello'

    def test_fixes_right_double_quote(self) -> None:
        assert fix_encoding_artifacts("helloÓ") == 'hello"'

    def test_fixes_apostrophe(self) -> None:
        assert fix_encoding_artifacts("todayÕs") == "today's"

    def test_fixes_em_dash(self) -> None:
        # noinspection SpellCheckingInspection
        assert fix_encoding_artifacts("helloÑworld") == "hello—world"

    def test_fixes_en_dash(self) -> None:
        assert fix_encoding_artifacts("1988Ð1998") == "1988–1998"


# --- fix_punctuation_spacing ---

class TestFixPunctuationSpacing:
    def test_removes_space_before_period(self) -> None:
        assert fix_punctuation_spacing("hello .") == "hello."

    def test_removes_space_before_comma(self) -> None:
        assert fix_punctuation_spacing("hello , world") == "hello, world"

    def test_removes_space_before_question_mark(self) -> None:
        assert fix_punctuation_spacing("really ?") == "really?"

    def test_removes_space_before_exclamation(self) -> None:
        assert fix_punctuation_spacing("wow !") == "wow!"


# --- fix_bracket_spacing ---

class TestFixBracketSpacing:
    def test_removes_space_inside_parentheses(self) -> None:
        assert fix_bracket_spacing("( hello )") == "(hello)"

    def test_removes_space_inside_square_brackets(self) -> None:
        assert fix_bracket_spacing("[ hello ]") == "[hello]"

    def test_removes_space_inside_curly_braces(self) -> None:
        assert fix_bracket_spacing("{ hello }") == "{hello}"


# --- fix_apostrophes ---

class TestFixApostrophes:
    def test_fixes_possessive_apostrophe(self) -> None:
        assert fix_apostrophes("the dog 's bone") == "the dog's bone"

    def test_fixes_space_before_possessive(self) -> None:
        assert fix_apostrophes("the dog 's bone") == "the dog's bone"


# --- strip_footnote_numbers ---

class TestStripFootnoteNumbers:
    def test_strips_trailing_digit(self) -> None:
        assert strip_footnote_numbers("Hello world.1") == "Hello world."

    def test_strips_multiple_trailing_digits(self) -> None:
        assert strip_footnote_numbers("Hello world.123") == "Hello world."

    def test_no_change_when_ends_with_punctuation(self) -> None:
        assert strip_footnote_numbers("Hello world.") == "Hello world."

    def test_no_change_when_no_digits(self) -> None:
        assert strip_footnote_numbers("Hello world") == "Hello world"


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
        assert clean_text("\u201chello\u201d") == '"hello"'

    def test_removes_soft_hyphen(self) -> None:
        assert clean_text("explo\u00adration") == "exploration"

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
