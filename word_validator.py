import re
from typing import Union


class WordValidator:
    """Validates and normalises words using NLTK stemming and lemmatization.

    Lazily loads NLTK resources on first use and caches them for the
    lifetime of the instance. Provides methods to check whether a word
    is a valid English word and to combine hyphenated words in text.

    Attributes:
        _words_list: Cached set of English words from the NLTK corpus.
        _lemmatizer: Cached WordNetLemmatizer instance.
        _stemmer: Cached PorterStemmer instance.
    """

    def __init__(self) -> None:
        """Initialise WordValidator with empty caches."""
        self._words_list: set | None = None
        self._lemmatizer = None
        self._stemmer = None

    def _get_words_list(self) -> set:
        """Lazily load and cache the NLTK words list."""
        if self._words_list is None:
            import nltk
            nltk.download('words')
            self._words_list = set(nltk.corpus.words.words())
        return self._words_list

    def _get_lemmatizer(self):
        """Lazily load and cache the WordNetLemmatizer."""
        if self._lemmatizer is None:
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    def _get_stemmer(self):
        """Lazily load and cache the PorterStemmer."""
        if self._stemmer is None:
            from nltk.stem import PorterStemmer
            self._stemmer = PorterStemmer()
        return self._stemmer

    def is_valid_word(self, word: str) -> Union[bool, str]:
        """Check if a word is valid by comparing it directly and via stemming/lemmatization.

        Returns True (or the valid modified word) if the word is found,
        otherwise returns False.

        Args:
            word: The word to validate.

        Returns:
            True or the valid modified form of the word if valid, False otherwise.
        """
        words_list = self._get_words_list()
        stemmer = self._get_stemmer()
        lemmatizer = self._get_lemmatizer()

        stem = stemmer.stem(word)
        if word.lower() in words_list or word in words_list:
            return True
        elif stem in words_list or stem.lower() in words_list:
            return True

        # Check all lemmatizations of the word
        for pos in ['n', 'v', 'a', 'r', 's']:
            lemma = lemmatizer.lemmatize(word, pos=pos)
            if lemma in words_list:
                return True

        # Check for custom lemmatizations
        # noinspection SpellCheckingInspection
        suffixes = {
            "ability": "able",  # testability -> testable
            "ibility": "ible",  # possibility -> possible
            "iness": "y",  # happiness -> happy
            "ity": "e",  # creativity -> create
            "tion": "e",  # creation -> create
            "able": "",  # testable -> test
            "ible": "",  # possible -> poss
            "ing": "",  # running -> run
            "ed": "",  # tested -> test
            "s": ""  # tests -> test
        }
        for suffix, replacement in suffixes.items():
            if word.endswith(suffix):
                stripped_word = word[:-len(suffix)] + replacement
                # Recursively check the modified word; if valid, return the modified form.
                result = self.is_valid_word(stripped_word)
                if result:
                    return result

        return False

    def combine_hyphenated_words(self, p_str: str) -> str:
        """Combine hyphenated words if the parts together form a valid word.

        Otherwise, preserve the hyphen (assuming it connects two valid words).

        Args:
            p_str: The input string potentially containing hyphenated words.

        Returns:
            The string with hyphens resolved appropriately.
        """

        def replace_dash(match) -> str:
            word1, word2 = match.group(1), match.group(2)
            combined = word1.strip() + word2.strip()

            # If there is a space after the hyphen and the combined word is valid,
            # assume the hyphen was splitting a single word.
            if word2.startswith(" ") and self.is_valid_word(combined):
                return combined
            # If both parts are valid words on their own, keep them hyphenated.
            elif self.is_valid_word(word1.strip()) and self.is_valid_word(word2.strip()):
                return word1.strip() + '-' + word2.strip()
            # Otherwise, if the combined word is valid, return it.
            elif self.is_valid_word(combined):
                return combined
            # If the combined word starts with a capital letter (likely a proper noun)
            # and the second part isn't valid on its own, combine them.
            elif combined[0].isupper() and not word2.strip()[0].isupper() and not self.is_valid_word(word2.strip()):
                return combined

            # Default: assume the hyphen is meant to connect two words.
            return word1.strip() + '-' + word2.strip()

        # Replace any soft hyphen characters with a regular dash.
        p_str = p_str.replace("­", "-")
        # Look for hyphens between word parts (with or without an extra space)
        p_str = re.sub(r'(\w+)-(\s?\w+)', replace_dash, p_str)

        return p_str


# Module-level instance for use across the codebase
_word_validator = WordValidator()
