import pytest
from unittest.mock import patch
from text_cleaner import TextCleaner

patch_llm_chat: str = 'text_cleaner.ollama.chat'


# --- Fixtures ---

def make_cleaner(model: str = 'llama3.1:8b', max_retries: int = 3) -> TextCleaner:
    """Create a TextCleaner instance."""
    return TextCleaner(model=model, max_retries=max_retries)


def make_response(cleaned: str, classification: str) -> dict:
    """Create a mock LLM response."""
    return {
        'message': {
            'content': f'{{"cleaned": "{cleaned}", "classification": "{classification}"}}'
        }
    }


# --- TestClean ---

class TestClean:
    def test_body_classification(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Clean body text.", "body")):
            cleaned, classification = cleaner.clean("Clean body text.")
        assert cleaned == "Clean body text."
        assert classification == "body"

    def test_footnote_classification(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("A footnote.", "footnote")):
            cleaned, classification = cleaner.clean("A footnote.1")
        assert cleaned == "A footnote."
        assert classification == "footnote"

    def test_drop_classification(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Chapter 1", "drop")):
            cleaned, classification = cleaner.clean("Chapter 1")
        assert classification == "drop"

    def test_cleaned_text_returned_correctly(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Fixed text.", "body")):
            cleaned, _ = cleaner.clean("Brok en text.")
        assert cleaned == "Fixed text."

    def test_empty_paragraph(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("", "drop")):
            cleaned, classification = cleaner.clean("")
        assert cleaned == ""
        assert classification == "drop"

    def test_uses_configured_model(self):
        cleaner = make_cleaner(model='llama3.2:3b')
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Text.")
        assert mock_chat.call_args[1]['model'] == 'llama3.2:3b'

    def test_passes_paragraph_as_user_message_without_page_context(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Some paragraph text.")
        messages = mock_chat.call_args[1]['messages']
        user_message = next(m for m in messages if m['role'] == 'user')
        assert user_message['content'] == "Paragraph to clean and classify:\nSome paragraph text."

    def test_includes_page_context_in_user_message(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Current paragraph.", page_context="Full page text here.")
        messages = mock_chat.call_args[1]['messages']
        user_message = next(m for m in messages if m['role'] == 'user')
        assert "Full page text here." in user_message['content']
        assert "Current paragraph." in user_message['content']

    def test_page_context_not_included_when_empty(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Some paragraph text.", page_context="")
        messages = mock_chat.call_args[1]['messages']
        user_message = next(m for m in messages if m['role'] == 'user')
        assert user_message['content'] == "Paragraph to clean and classify:\nSome paragraph text."

    def test_page_context_not_included_when_not_provided(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Some paragraph text.")
        messages = mock_chat.call_args[1]['messages']
        user_message = next(m for m in messages if m['role'] == 'user')
        assert "Page context" not in user_message['content']

    def test_system_prompt_included(self):
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Text.")
        messages = mock_chat.call_args[1]['messages']
        system_message = next(m for m in messages if m['role'] == 'system')
        assert system_message['content']


# --- TestRetry ---

class TestRetry:
    def test_retries_on_malformed_json(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        good_response = make_response("Clean text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert cleaned == "Clean text."
        assert classification == "body"

    def test_retries_on_invalid_classification(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = make_response("Text.", "invalid_type")
        good_response = make_response("Text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert classification == "body"

    def test_retries_on_missing_key(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': '{"cleaned": "Text."}'}}
        good_response = make_response("Text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert classification == "body"

    def test_not_raises_after_max_retries_exceeded(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        with patch(patch_llm_chat, return_value=bad_response):
            cleaned, classification = cleaner.clean("Some text.")
            assert cleaned == ""
            assert classification == 'body'

    def test_correct_number_of_attempts(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        with patch(patch_llm_chat, return_value=bad_response) as mock_chat:
            cleaner.clean("Some text.")
        assert mock_chat.call_count == 3

    def test_succeeds_on_last_retry(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        good_response = make_response("Clean text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert cleaned == "Clean text."
        assert classification == "body"

    def test_max_retries_configurable(self):
        cleaner = make_cleaner(max_retries=5)
        bad_response = {'message': {'content': 'not valid json'}}
        with patch(patch_llm_chat, return_value=bad_response) as mock_chat:
            cleaner.clean("Some text.")
        assert mock_chat.call_count == 5

    def test_retry_uses_same_messages(self):
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        good_response = make_response("Text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]) as mock_chat:
            cleaner.clean("Some text.", page_context="Full page text.")
        for call in mock_chat.call_args_list:
            messages = call[1]['messages']
            user_message = next(m for m in messages if m['role'] == 'user')
            assert "Full page text." in user_message['content']
            assert "Some text." in user_message['content']


# --- TestIntegration ---

class TestIntegration:
    @pytest.mark.integration
    def test_real_llm_call_body(self):
        """Integration test — requires a running LLM with llama3.1:8b."""
        cleaner = make_cleaner()
        paragraph = "This is a sample paragraph from a book about philosophy and rationality."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert "philosophy" in cleaned
        assert "rationality" in cleaned
        assert cleaned == paragraph

    @pytest.mark.integration
    def test_real_llm_call_footnote(self):
        """Integration test — requires a running LLM with llama3.1:8b."""
        cleaner = make_cleaner()
        page_context = (
            "Others have found very similar defection rates in various minor religious sects.1\n\n"
            "1 This ignores the interesting question of whether the defectors have given up "
            "all the beliefs in the doctrines of the movement they have quit."
        )
        cleaned, classification = cleaner.clean(
            "1 This ignores the interesting question of whether the defectors have given up "
            "all the beliefs in the doctrines of the movement they have quit.",
            page_context=page_context
        )
        assert classification in ('footnote', 'drop')
        assert not cleaned.startswith("1 ")
        assert cleaned == ("This ignores the interesting question of whether the defectors have given up "
                           "all the beliefs in the doctrines of the movement they have quit.")

    @pytest.mark.integration
    def test_real_llm_call_drop(self):
        """Integration test — requires a running LLM with llama3.1:8b."""
        cleaner = make_cleaner()
        cleaned, classification = cleaner.clean(
            "Chapter 1 ... 1\nChapter 2 ... 15\nChapter 3 ... 42"
        )
        assert classification == 'drop'

    @pytest.mark.integration
    def test_real_llm_call_body_unchanged(self):
        """Clean prose with no issues should be returned exactly as-is."""
        cleaner = make_cleaner()
        paragraph = "The French Revolution began in 1789 and fundamentally transformed the political landscape of Europe."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert cleaned == paragraph

    @pytest.mark.integration
    def test_real_llm_call_ocr_word_break_fixed(self):
        """Mid-word line breaks introduced by OCR should be rejoined."""
        cleaner = make_cleaner()
        paragraph = "The development of mod- ern philosophy can be traced to the six- teenth century."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert cleaned == "The development of modern philosophy can be traced to the sixteenth century."

    @pytest.mark.integration
    def test_real_llm_call_trailing_footnote_marker_stripped(self):
        """A trailing footnote number at the end of a body paragraph should be removed."""
        cleaner = make_cleaner()
        paragraph = "The movement grew rapidly throughout the nineteenth century, attracting followers from across the social spectrum. 4"
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert cleaned == "The movement grew rapidly throughout the nineteenth century, attracting followers from across the social spectrum."

    @pytest.mark.integration
    def test_real_llm_call_footnote_identified_with_page_context(self):
        """A footnote paragraph should be identified and its leading number stripped when page context is provided."""
        cleaner = make_cleaner()
        page_context = (
            "The movement grew rapidly throughout the nineteenth century, "
            "attracting followers from across the social spectrum.4\n\n"
            "4 For full membership statistics by region, see Jones (1987), pp. 142-156."
        )
        paragraph = "4 For full membership statistics by region, see Jones (1987), pp. 142-156."
        cleaned, classification = cleaner.clean(paragraph, page_context=page_context)
        assert classification == 'footnote'
        assert cleaned == "For full membership statistics by region, see Jones (1987), pp. 142-156."

    @pytest.mark.integration
    def test_real_llm_call_footnote_without_page_context(self):
        """Without page context the response should still be valid, even if classification varies."""
        cleaner = make_cleaner()
        paragraph = "4 For full membership statistics by region, see Jones (1987), pp. 142-156."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification in ('body', 'footnote', 'drop')
        assert isinstance(cleaned, str)

    @pytest.mark.integration
    def test_real_llm_call_drop_toc(self):
        """An obvious table of contents should be classified as drop."""
        cleaner = make_cleaner()
        cleaned, classification = cleaner.clean(
            "Introduction ... 1\nChapter One: The Early Years ... 15\n"
            "Chapter Two: The Middle Period ... 47\nConclusion ... 203"
        )
        assert classification == 'drop'
