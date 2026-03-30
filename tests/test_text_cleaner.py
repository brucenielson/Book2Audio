import pytest
from unittest.mock import MagicMock, patch
from text_cleaner import TextCleaner

patch_ollama_chat: str = 'text_cleaner.ollama.chat'


# --- Fixtures ---

def make_cleaner(model: str = 'llama3.1:8b', max_retries: int = 3) -> TextCleaner:
    """Create a TextCleaner instance."""
    return TextCleaner(model=model, max_retries=max_retries)


def make_response(cleaned: str, classification: str) -> dict:
    """Create a mock ollama.chat response."""
    return {
        'message': {
            'content': f'{{"cleaned": "{cleaned}", "classification": "{classification}"}}'
        }
    }

# --- TestClean ---

class TestClean:
    def test_body_classification(self):
        cleaner = make_cleaner()
        with patch(patch_ollama_chat, return_value=make_response("Clean body text.", "body")):
            cleaned, classification = cleaner.clean("Clean body text.")
        assert cleaned == "Clean body text."
        assert classification == "body"

    def test_footnote_classification(self):
        cleaner = make_cleaner()
        with patch(patch_ollama_chat, return_value=make_response("A footnote.", "footnote")):
            cleaned, classification = cleaner.clean("A footnote.1")
        assert cleaned == "A footnote."
        assert classification == "footnote"

    def test_drop_classification(self):
        cleaner = make_cleaner()
        with patch(patch_ollama_chat, return_value=make_response("Chapter 1", "drop")):
            cleaned, classification = cleaner.clean("Chapter 1")
        assert classification == "drop"

    def test_cleaned_text_returned_correctly(self):
        cleaner = make_cleaner()
        with patch(patch_ollama_chat, return_value=make_response("Fixed text.", "body")):
            cleaned, _ = cleaner.clean("Brok en text.")
        assert cleaned == "Fixed text."

    def test_empty_paragraph(self):
        cleaner = make_cleaner()
        with patch(patch_ollama_chat, return_value=make_response("", "drop")):
            cleaned, classification = cleaner.clean("")
        assert cleaned == ""
        assert classification == "drop"

    def test_uses_configured_model(self):
        cleaner = make_cleaner(model='llama3.2:3b')
        with patch(patch_ollama_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Text.")
        assert mock_chat.call_args[1]['model'] == 'llama3.2:3b'

    def test_passes_paragraph_as_user_message(self):
        cleaner = make_cleaner()
        with patch(patch_ollama_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Some paragraph text.")
        messages = mock_chat.call_args[1]['messages']
        user_message = next(m for m in messages if m['role'] == 'user')
        assert user_message['content'] == "Some paragraph text."

    def test_system_prompt_included(self):
        cleaner = make_cleaner()
        with patch(patch_ollama_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Text.")
        messages = mock_chat.call_args[1]['messages']
        system_message = next(m for m in messages if m['role'] == 'system')
        assert system_message['content']


# --- TestRetry ---

class TestRetry:
    def test_retries_on_malformed_json(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = MagicMock()
        bad_response.__getitem__ = MagicMock(side_effect=lambda k: {
            'message': {'content': 'not valid json'}
        }[k])
        good_response = make_response("Clean text.", "body")
        with patch(patch_ollama_chat, side_effect=[bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert cleaned == "Clean text."
        assert classification == "body"

    def test_retries_on_invalid_classification(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = make_response("Text.", "invalid_type")
        good_response = make_response("Text.", "body")
        with patch(patch_ollama_chat, side_effect=[bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert classification == "body"

    def test_retries_on_missing_key(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = MagicMock()
        bad_response.__getitem__ = MagicMock(side_effect=lambda k: {
            'message': {'content': '{"cleaned": "Text."}'}
        }[k])
        good_response = make_response("Text.", "body")
        with patch(patch_ollama_chat, side_effect=[bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert classification == "body"

    def test_raises_after_max_retries_exceeded(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = MagicMock()
        bad_response.__getitem__ = MagicMock(side_effect=lambda k: {
            'message': {'content': 'not valid json'}
        }[k])
        with patch(patch_ollama_chat, return_value=bad_response):
            with pytest.raises(ValueError):
                cleaner.clean("Some text.")

    def test_correct_number_of_attempts(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = MagicMock()
        bad_response.__getitem__ = MagicMock(side_effect=lambda k: {
            'message': {'content': 'not valid json'}
        }[k])
        with patch(patch_ollama_chat, return_value=bad_response) as mock_chat:
            with pytest.raises(ValueError):
                cleaner.clean("Some text.")
        assert mock_chat.call_count == 3

    def test_succeeds_on_last_retry(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = MagicMock()
        bad_response.__getitem__ = MagicMock(side_effect=lambda k: {
            'message': {'content': 'not valid json'}
        }[k])
        good_response = make_response("Clean text.", "body")
        with patch(patch_ollama_chat, side_effect=[bad_response, bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert cleaned == "Clean text."
        assert classification == "body"

    def test_max_retries_configurable(self):
        cleaner = make_cleaner(max_retries=5)
        bad_response = MagicMock()
        bad_response.__getitem__ = MagicMock(side_effect=lambda k: {
            'message': {'content': 'not valid json'}
        }[k])
        with patch(patch_ollama_chat, return_value=bad_response) as mock_chat:
            with pytest.raises(ValueError):
                cleaner.clean("Some text.")
        assert mock_chat.call_count == 5


# --- TestIntegration ---

class TestIntegration:
    @pytest.mark.integration
    def test_real_ollama_call_body(self):
        """Integration test — requires Ollama running with llama3.1:8b."""
        cleaner = make_cleaner()
        cleaned, classification = cleaner.clean(
            "This is a sample paragraph from a book about philosophy and rationality."
        )
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
        assert classification in ('body', 'footnote', 'drop')

    @pytest.mark.integration
    def test_real_ollama_call_footnote(self):
        """Integration test — requires Ollama running with llama3.1:8b."""
        cleaner = make_cleaner()
        cleaned, classification = cleaner.clean(
            "1. See also Smith (1984) for a detailed discussion of this topic."
        )
        assert classification in ('footnote', 'drop')

    @pytest.mark.integration
    def test_real_ollama_call_drop(self):
        """Integration test — requires Ollama running with llama3.1:8b."""
        cleaner = make_cleaner()
        cleaned, classification = cleaner.clean(
            "Chapter 1 ... 1\nChapter 2 ... 15\nChapter 3 ... 42"
        )
        assert classification == 'drop'
