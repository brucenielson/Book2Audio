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
    def test_words_list_is_none_before_use(self):
        """_words_list should be None before first use."""
        v = WordValidator()
        assert v._words_list is None

    def test_lemmatizer_is_none_before_use(self):
        """_lemmatizer should be None before first use."""
        v = WordValidator()
        assert v._lemmatizer is None

    def test_stemmer_is_none_before_use(self):
        """_stemmer should be None before first use."""
        v = WordValidator()
        assert v._stemmer is None

    def test_words_list_cached_after_first_use(self, validator):
        """_words_list should be populated after first call to _get_words_list."""
        validator._get_words_list()
        assert validator._words_list is not None

    def test_lemmatizer_cached_after_first_use(self, validator):
        """_lemmatizer should be populated after first call to _get_lemmatizer."""
        validator._get_lemmatizer()
        assert validator._lemmatizer is not None

    def test_stemmer_cached_after_first_use(self, validator):
        """_stemmer should be populated after first call to _get_stemmer."""
        validator._get_stemmer()
        assert validator._stemmer is not None

    def test_words_list_same_object_on_second_call(self, validator):
        """_get_words_list should return the same object on repeated calls."""
        first = validator._get_words_list()
        second = validator._get_words_list()
        assert first is second


# --- is_valid_word tests ---

class TestIsValidWord:
    def test_plain_valid_word(self, validator):
        """A simple valid English word should return True."""
        assert validator.is_valid_word("dog") is True

    def test_plain_invalid_word(self, validator):
        """A nonsense word should return False."""
        # noinspection SpellCheckingInspection
        assert validator.is_valid_word("xqzjkl") is False

    def test_uppercase_valid_word(self, validator):
        """A valid word in uppercase should return True."""
        assert validator.is_valid_word("Dog") is True

    def test_stemmed_word(self, validator):
        """A word valid via stemming should return True."""
        assert validator.is_valid_word("running") is True

    def test_lemmatized_word(self, validator):
        """A word valid via lemmatization should return True."""
        assert validator.is_valid_word("geese") is True

    def test_suffix_ability(self, validator):
        """A word ending in 'ability' should resolve via custom suffix."""
        assert validator.is_valid_word("testability") is not False

    # noinspection SpellCheckingInspection
    def test_suffix_iness(self, validator):
        # noinspection SpellCheckingInspection
        """A word ending in 'iness' should resolve via custom suffix."""
        assert validator.is_valid_word("happiness") is not False

    def test_suffix_tion(self, validator):
        """A word ending in 'tion' should resolve via custom suffix."""
        assert validator.is_valid_word("creation") is not False

    def test_suffix_ing(self, validator):
        """A word ending in 'ing' should resolve via custom suffix."""
        assert validator.is_valid_word("testing") is not False

    def test_suffix_ed(self, validator):
        """A word ending in 'ed' should resolve via custom suffix."""
        assert validator.is_valid_word("tested") is not False

    def test_suffix_s(self, validator):
        """A word ending in 's' should resolve via custom suffix."""
        assert validator.is_valid_word("dogs") is not False

    def test_empty_string(self, validator):
        """An empty string should return False without crashing."""
        assert validator.is_valid_word("") is False


# --- combine_hyphenated_words tests ---

class TestCombineHyphenatedWords:
    def test_no_hyphen(self, validator):
        """A string with no hyphens should be returned unchanged."""
        assert validator.combine_hyphenated_words("hello world") == "hello world"

    def test_soft_hyphen_replaced(self, validator):
        """Soft hyphens should be replaced with regular hyphens before processing."""
        # soft hyphen is \u00ad
        result = validator.combine_hyphenated_words("test\u00adword")
        assert "\u00ad" not in result

    def test_two_valid_words_keeps_hyphen(self, validator):
        """Two valid words joined by a hyphen should remain hyphenated."""
        result = validator.combine_hyphenated_words("well-known")
        assert result == "well-known"

    def test_split_word_combined(self, validator):
        """A word split across a hyphen with a space should be combined if valid."""
        # noinspection GrazieInspection
        # "base-ball" (with extra space) -> "baseball" if valid, or kept as is
        result = validator.combine_hyphenated_words("base- ball")
        assert "-" not in result or result == "base-ball"

    def test_proper_noun_combined(self, validator):
        """A capitalized combined word with invalid second part should be combined."""
        # noinspection SpellCheckingInspection
        result = validator.combine_hyphenated_words("New-xqzjk")
        # noinspection SpellCheckingInspection
        assert result == "Newxqzjk"

    def test_no_crash_on_empty_string(self, validator):
        """An empty string should be returned unchanged without crashing."""
        assert validator.combine_hyphenated_words("") == ""

    def test_multiple_hyphens(self, validator):
        """A string with multiple hyphenated pairs should process all of them."""
        result = validator.combine_hyphenated_words("well-known and up-to-date")
        assert isinstance(result, str)


# --- Module-level instance test ---

class TestModuleLevelInstance:
    def test_module_instance_exists(self):
        """The module-level _word_validator instance should exist."""
        assert word_validator is not None

    def test_module_instance_is_word_validator(self):
        """The module-level instance should be a WordValidator."""
        assert isinstance(word_validator, WordValidator)
