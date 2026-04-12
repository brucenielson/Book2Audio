from nltk.corpus import words as nltk_words

_english_words: set[str] | None = None


def get_english_words() -> set[str]:
    """Lazily load and return the NLTK English word set.

    The set is built on first call and cached for subsequent calls.
    """
    global _english_words
    if _english_words is None:
        _english_words = set(w.lower() for w in nltk_words.words())
    return _english_words
