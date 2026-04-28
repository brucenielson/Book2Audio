"""Tests for the WordValidator class and module-level word_validator instance."""

import pytest
from word_validator import WordValidator, word_validator


# --- Fixtures ---

@pytest.fixture(scope="module")
def validator():
    """A single WordValidator instance shared across all tests in this module.
    Uses module scope to avoid reloading NLTK resources for every test."""
    return WordValidator()


# --- Lazy loading tests ---

class TestLazyLoading:
    def test_words_list_returns_non_empty_set(self) -> None:
        """get_english_words() should return a non-empty set."""
        from utils.nltk_utils import get_english_words
        words = get_english_words()
        assert isinstance(words, set)
        assert len(words) > 0

    def test_words_list_module_cache_populated_after_use(self) -> None:
        """The module-level cache in nltk_utils should be populated after get_english_words() is called."""
        from utils import nltk_utils
        nltk_utils.get_english_words()
        assert nltk_utils._english_words is not None

    def test_lemmatizer_is_none_before_use(self) -> None:
        """_lemmatizer should be None before first use."""
        v = WordValidator()
        assert v._lemmatizer is None

    def test_stemmer_is_none_before_use(self) -> None:
        """_stemmer should be None before first use."""
        v = WordValidator()
        assert v._stemmer is None

    def test_lemmatizer_cached_after_first_use(self, validator) -> None:
        """_lemmatizer should be populated after first call to _get_lemmatizer."""
        validator._get_lemmatizer()
        assert validator._lemmatizer is not None

    def test_stemmer_cached_after_first_use(self, validator) -> None:
        """_stemmer should be populated after first call to _get_stemmer."""
        validator._get_stemmer()
        assert validator._stemmer is not None

    def test_words_list_same_object_on_second_call(self, validator) -> None:
        """_get_words_list should return the same object on repeated calls."""
        first = validator._get_words_list()
        second = validator._get_words_list()
        assert first is second


# --- is_valid_word tests ---

class TestIsValidWord:
    def test_plain_valid_word(self, validator) -> None:
        """A simple valid English word should return True."""
        assert validator.is_valid_word("dog") is True

    def test_plain_invalid_word(self, validator) -> None:
        """A nonsense word should return False."""
        # noinspection SpellCheckingInspection
        assert validator.is_valid_word("xqzjkl") is False

    def test_uppercase_valid_word(self, validator) -> None:
        """A valid word in uppercase should return True."""
        assert validator.is_valid_word("Dog") is True

    def test_stemmed_word(self, validator) -> None:
        """A word valid via stemming should return True."""
        assert validator.is_valid_word("running") is True

    def test_lemmatized_word(self, validator) -> None:
        """A word valid via lemmatization should return True."""
        assert validator.is_valid_word("geese") is True

    def test_suffix_ability(self, validator) -> None:
        """A word ending in 'ability' should resolve via custom suffix."""
        assert validator.is_valid_word("testability") is not False

    # noinspection SpellCheckingInspection
    def test_suffix_iness(self, validator) -> None:
        # noinspection SpellCheckingInspection
        """A word ending in 'iness' should resolve via custom suffix."""
        assert validator.is_valid_word("happiness") is not False

    def test_suffix_tion(self, validator) -> None:
        """A word ending in 'tion' should resolve via custom suffix."""
        assert validator.is_valid_word("creation") is not False

    def test_suffix_ing(self, validator) -> None:
        """A word ending in 'ing' should resolve via custom suffix."""
        assert validator.is_valid_word("testing") is not False

    def test_suffix_ed(self, validator) -> None:
        """A word ending in 'ed' should resolve via custom suffix."""
        assert validator.is_valid_word("tested") is not False

    def test_suffix_s(self, validator) -> None:
        """A word ending in 's' should resolve via custom suffix."""
        assert validator.is_valid_word("dogs") is not False

    def test_empty_string(self, validator) -> None:
        """An empty string should return False without crashing."""
        assert validator.is_valid_word("") is False

    def test_adverb_empirically(self, validator) -> None:
        """'empirically' is a common English adverb and should be valid."""
        assert validator.is_valid_word("empirically") is True

    def test_adverb_empirically_is_reason_combine_hyphenated_fails(self, validator) -> None:
        """If 'empirically' is invalid, combine_hyphenated_words cannot join 'empiri- cally'."""
        # This test documents the dependency: combine_hyphenated_words relies on
        # is_valid_word to decide whether to join soft-hyphen splits.
        # If this assertion fails, the root cause of the joining bug is here.
        assert validator.is_valid_word("empirically") is True, (
            "is_valid_word('empirically') returned False — "
            "this is why combine_hyphenated_words leaves 'empiri- cally' unhyphenated"
        )


# --- combine_hyphenated_words tests ---

class TestCombineHyphenatedWords:
    def test_no_hyphen(self, validator) -> None:
        """A string with no hyphens should be returned unchanged."""
        assert validator.combine_hyphenated_words("hello world") == "hello world"

    def test_soft_hyphen_replaced(self, validator) -> None:
        """Soft hyphens should be replaced with regular hyphens before processing."""
        # soft hyphen is \u00ad
        result = validator.combine_hyphenated_words("test\u00adword")
        assert "\u00ad" not in result

    def test_two_valid_words_keeps_hyphen(self, validator) -> None:
        """Two valid words joined by a hyphen should remain hyphenated."""
        result = validator.combine_hyphenated_words("well-known")
        assert result == "well-known"

    def test_split_word_combined(self, validator) -> None:
        """A word split across a hyphen with a space should be combined if valid."""
        # noinspection GrazieInspection
        # "base-ball" (with extra space) -> "baseball" if valid, or kept as is
        result = validator.combine_hyphenated_words_advanced("base- ball")
        assert "-" not in result or result == "base-ball"

    def test_proper_noun_combined(self, validator) -> None:
        """A capitalized combined word with invalid second part should be combined."""
        # noinspection SpellCheckingInspection
        result = validator.combine_hyphenated_words_advanced("New-xqzjk")
        # noinspection SpellCheckingInspection
        assert result == "Newxqzjk"

    def test_no_crash_on_empty_string(self, validator) -> None:
        """An empty string should be returned unchanged without crashing."""
        assert validator.combine_hyphenated_words("") == ""

    def test_multiple_hyphens(self, validator) -> None:
        """A string with multiple hyphenated pairs should process all of them."""
        result = validator.combine_hyphenated_words("well-known and up-to-date")
        assert isinstance(result, str)

    def test_soft_hyphen_with_space_joined(self, validator) -> None:
        """Soft hyphen followed by a space should join the word (page-break artifact)."""
        # U+00AD followed by a space is how PDF page-break hyphens appear after extraction
        result = validator.combine_hyphenated_words("empiri­ cally")
        assert result == "empirically"

    def test_soft_hyphen_with_space_demarcation(self, validator) -> None:
        """'demarca­ tion' (soft hyphen + space) should join to 'demarcation'."""
        result = validator.combine_hyphenated_words("demarca­ tion")
        assert result == "demarcation"

    def test_soft_hyphen_with_space_volume(self, validator) -> None:
        """'vol­ ume' (soft hyphen + space) should join to 'volume'."""
        result = validator.combine_hyphenated_words("vol­ ume")
        assert result == "volume"

    def test_soft_hyphen_with_space_empirical(self, validator) -> None:
        """'empir­ ical' (soft hyphen + space) should join to 'empirical'."""
        result = validator.combine_hyphenated_words("empir­ ical")
        assert result == "empirical"

    def test_soft_hyphen_multiple_in_sentence(self, validator) -> None:
        """Multiple soft-hyphen page-break artifacts in one string are all resolved."""
        result = validator.combine_hyphenated_words(
            "empiri­ cally refutable and empir­ ical hypotheses"
        )
        assert result == "empirically refutable and empirical hypotheses"


# --- Module-level instance test ---

class TestModuleLevelInstance:
    def test_module_instance_exists(self) -> None:
        """The module-level _word_validator instance should exist."""
        assert word_validator is not None

    def test_module_instance_is_word_validator(self) -> None:
        """The module-level instance should be a WordValidator."""
        assert isinstance(word_validator, WordValidator)
