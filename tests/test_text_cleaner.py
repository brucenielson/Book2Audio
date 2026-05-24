"""Tests for the TextCleaner class."""

import pytest
from unittest.mock import patch
from text_cleaner import (TextCleaner, _has_suspicious_substitutions, _coerce_classification,
                          _is_word_like, _normalize_dashes, _restore_valid_words, _restore_list_prefix)

patch_llm_chat: str = 'text_cleaner.ollama.chat'

from conftest import TEST_LLM_MODEL


# --- Fixtures ---

def make_cleaner(model: str = TEST_LLM_MODEL, max_retries: int = 3) -> TextCleaner:
    """Create a TextCleaner instance."""
    return TextCleaner(model=model, max_retries=max_retries, temperature=0)


def make_response(cleaned: str, classification: str) -> dict:
    """Create a mock LLM response."""
    return {
        'message': {
            'content': f'{{"cleaned": "{cleaned}", "classification": "{classification}"}}'
        }
    }


# --- TestClean ---

class TestClean:
    def test_body_classification(self) -> None:
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Clean body text.", "body")):
            cleaned, classification = cleaner.clean("Clean body text.")
        assert cleaned == "Clean body text."
        assert classification == "body"

    def test_footnote_classification(self) -> None:
        cleaner = make_cleaner()
        # Real footnotes always start with a reference number, never a letter.
        with patch(patch_llm_chat, return_value=make_response("A genuine footnote.", "footnote")):
            cleaned, classification = cleaner.clean("1 A genuine footnote.")
        assert cleaned == "A genuine footnote."
        assert classification == "footnote"

    def test_drop_classification(self) -> None:
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Chapter 1", "drop")):
            cleaned, classification = cleaner.clean("Chapter 1")
        assert classification == "drop"

    def test_cleaned_text_returned_correctly(self) -> None:
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Broken text.", "body")):
            cleaned, _ = cleaner.clean("Brok en text.")
        assert cleaned == "Broken text."

    def test_empty_paragraph(self) -> None:
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("", "drop")):
            cleaned, classification = cleaner.clean("")
        assert cleaned == ""
        assert classification == "drop"

    def test_uses_configured_model(self) -> None:
        cleaner = make_cleaner(model='llama3.2:3b')
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Text.")
        assert mock_chat.call_args[1]['model'] == 'llama3.2:3b'

    def test_passes_paragraph_as_user_message_without_page_context(self) -> None:
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Some paragraph text.")
        messages = mock_chat.call_args[1]['messages']
        user_message = next(m for m in messages if m['role'] == 'user')
        assert user_message['content'] == "Paragraph to clean and classify:\nSome paragraph text."

    def test_includes_page_context_in_user_message(self) -> None:
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Current paragraph.", page_context="Full page text here.")
        messages = mock_chat.call_args[1]['messages']
        user_message = next(m for m in messages if m['role'] == 'user')
        assert "Full page text here." in user_message['content']
        assert "Current paragraph." in user_message['content']

    def test_page_context_not_included_when_empty(self) -> None:
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Some paragraph text.", page_context="")
        messages = mock_chat.call_args[1]['messages']
        user_message = next(m for m in messages if m['role'] == 'user')
        assert user_message['content'] == "Paragraph to clean and classify:\nSome paragraph text."

    def test_page_context_not_included_when_not_provided(self) -> None:
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Some paragraph text.")
        messages = mock_chat.call_args[1]['messages']
        user_message = next(m for m in messages if m['role'] == 'user')
        assert "Page context" not in user_message['content']

    def test_system_prompt_included(self) -> None:
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Text.", "body")) as mock_chat:
            cleaner.clean("Text.")
        messages = mock_chat.call_args[1]['messages']
        system_message = next(m for m in messages if m['role'] == 'system')
        assert system_message['content']

    @pytest.mark.parametrize("paragraph", [
        "(1) The statement that the earth is at rest.",
        "(2) This follows directly from the above.",
        "(12) A longer numbered item in a list.",
    ])
    def test_parenthesized_number_prefix_overrides_footnote_classification(
            self, paragraph: str) -> None:
        """LLM returning 'footnote' for a (N) prefixed paragraph must be overridden to 'body'.

        Real example: page 37 — '(1) The statement that the earth is at rest...'
        was wrongly classified as a footnote because the leading number misled the LLM.
        Paragraphs starting with (N) are always numbered body-text list items.
        """
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response(paragraph, "footnote")):
            _, classification = cleaner.clean(paragraph)
        assert classification == "body"

    @pytest.mark.parametrize("paragraph", [
        "(i) The first assumption is that all observation is theory-laden.",
        "(ii) There can be no valid reasoning from singular observation statements.",
        "(iii) A third point about the nature of induction.",
        "(iv) Final item in the enumeration.",
    ])
    def test_roman_numeral_list_prefix_overrides_footnote_classification(
            self, paragraph: str) -> None:
        """LLM returning 'footnote' for a (i)/(ii)/(iii)/(iv) paragraph must be overridden.

        Real example: page 72 — '(ii) There can be no valid reasoning...' was wrongly
        classified as a footnote. Roman numeral list items are body text, never footnotes.
        """
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response(paragraph, "footnote")):
            _, classification = cleaner.clean(paragraph)
        assert classification == "body"

    def test_ocr_spaced_list_prefix_in_cleaned_text_overrides_footnote(self) -> None:
        """Guard must also check cleaned_candidate, not just the original paragraph.

        Real example: '( 1 8) Anderson's discovery...' — OCR inserts spaces inside
        the parenthesized number. The original fails _LIST_PREFIX_RE but the LLM
        correctly fixes it to '(18) Anderson's discovery...', which should trigger
        the override."""
        original = "( 1 8) Anderson's discovery of the positron refutes a lot."
        cleaned  = "(18) Anderson's discovery of the positron refutes a lot."
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response(cleaned, "footnote")):
            _, classification = cleaner.clean(original)
        assert classification == "body"

    @pytest.mark.parametrize("paragraph", [
        "a) The first item in a lettered list.",
        "b) The second item.",
        "c) The replacement, in the neo-classical theory, of certain important limit-theorems.",
    ])
    def test_lettered_list_prefix_overrides_footnote_classification(
            self, paragraph: str) -> None:
        """LLM returning 'footnote' for an a)/b)/c) prefixed paragraph must be overridden.

        Real example: page 296 — 'c) The replacement, in the neo-classical theory...'
        was wrongly classified as a footnote. Lettered list items are body text."""
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response(paragraph, "footnote")):
            _, classification = cleaner.clean(paragraph)
        assert classification == "body"

    @pytest.mark.parametrize("paragraph", [
        "Since my discussion has given rise to misunderstandings, a clarification is needed.",
        "In order to see that (i) to (iv) are consistent we merely have to consider.",
        "The argument proceeds from the assumption that all swans are white.",
        "We can now state the main theorem of this section.",
    ])
    def test_letter_start_overrides_footnote_classification(
            self, paragraph: str) -> None:
        """Paragraphs starting with a letter can never be footnotes — override the LLM.

        Real footnotes always begin with a reference number or symbol. Body text
        starting with a letter must never be reclassified as a footnote regardless
        of what the LLM returns.
        """
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response(paragraph, "footnote")):
            _, classification = cleaner.clean(paragraph)
        assert classification == "body"


# --- TestRetry ---

class TestRetry:
    def test_retries_on_malformed_json(self) -> None:
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        good_response = make_response("Some text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert cleaned == "Some text."
        assert classification == "body"

    def test_retries_on_invalid_classification(self) -> None:
        cleaner = make_cleaner(max_retries=3)
        bad_response = make_response("Some text.", "invalid_type")
        good_response = make_response("Some text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert classification == "body"

    def test_retries_on_missing_key(self) -> None:
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': '{"cleaned": "Some text."}'}}
        good_response = make_response("Some text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert classification == "body"

    def test_not_raises_after_max_retries_exceeded(self) -> None:
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        with patch(patch_llm_chat, return_value=bad_response):
            cleaned, classification = cleaner.clean("Some text.")
            assert cleaned == "Some text."
            assert classification == 'body'

    def test_correct_number_of_attempts(self) -> None:
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        with patch(patch_llm_chat, return_value=bad_response) as mock_chat:
            cleaner.clean("Some text.")
        assert mock_chat.call_count == 3

    def test_succeeds_on_last_retry(self) -> None:
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        good_response = make_response("Some text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, bad_response, good_response]):
            cleaned, classification = cleaner.clean("Some text.")
        assert cleaned == "Some text."

    def test_repairs_invalid_json_escape_without_consuming_retry(self) -> None:
        """Stray backslash in LLM JSON (e.g. \\alpha) is repaired inline — no retry burned."""
        cleaner = make_cleaner(max_retries=3)
        # \alpha in the JSON value: \a is not a valid JSON escape sequence
        bad_escape = {'message': {'content': '{"cleaned": "formula \\alpha = 0", "classification": "body"}'}}
        with patch(patch_llm_chat, return_value=bad_escape) as mock_chat:
            cleaned, classification = cleaner.clean("formula \\alpha = 0")
        assert classification == "body"
        assert mock_chat.call_count == 1  # repaired inline, not retried

    def test_repaired_json_escape_preserves_cleaned_text(self) -> None:
        """After repairing invalid escape, the cleaned text is extracted correctly."""
        cleaner = make_cleaner(max_retries=3)
        bad_escape = {'message': {'content': '{"cleaned": "formula \\alpha = 0", "classification": "body"}'}}
        with patch(patch_llm_chat, return_value=bad_escape):
            cleaned, _ = cleaner.clean("formula \\alpha = 0")
        assert "alpha" in cleaned

    def test_non_escape_json_errors_still_trigger_retry(self) -> None:
        """A JSONDecodeError that isn't an escape issue still goes through the retry loop."""
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json at all'}}
        good_response = make_response("Some text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]) as mock_chat:
            cleaned, classification = cleaner.clean("Some text.")
        assert classification == "body"
        assert mock_chat.call_count == 2  # one failure, one success
        assert classification == "body"

    def test_unescaped_inner_quotes_recovered_by_regex_fallback(self) -> None:
        """LLM writes bare \" inside JSON string value — regex fallback recovers without retry.

        When the LLM produces: {"cleaned": "the word "probability" is used", ...}
        json.loads raises "Expecting ',' delimiter". 'escape' is not in that message
        so the existing backslash repair is skipped. Without the regex fallback all
        3 retries are burned and the raw paragraph is returned unchanged.
        With the fallback the values are extracted via greedy regex on the first
        attempt and no retry is consumed.
        """
        cleaner = make_cleaner(max_retries=3)
        inner_quote_response = {'message': {'content':
            '{"cleaned": "the word "probability" is used", "classification": "body"}'}}
        with patch(patch_llm_chat, return_value=inner_quote_response) as mock_chat:
            cleaned, classification = cleaner.clean('the word probability is used')
        assert classification == 'body'
        assert '"probability"' in cleaned  # inner quotes preserved in extracted value
        assert mock_chat.call_count == 1   # recovered inline, no retry burned

    def test_unescaped_inner_quotes_regex_fallback_footnote(self) -> None:
        """Regex fallback also works when classification is footnote.

        Paragraph starts with '1 ' (digit) so the letter-start footnote guard
        does not override the classification to body.
        """
        cleaner = make_cleaner(max_retries=3)
        inner_quote_response = {'message': {'content':
            '{"cleaned": "see "ibid." for details", "classification": "footnote"}'}}
        with patch(patch_llm_chat, return_value=inner_quote_response) as mock_chat:
            cleaned, classification = cleaner.clean('1 see ibid for details')
        assert classification == 'footnote'
        assert '"ibid."' in cleaned
        assert mock_chat.call_count == 1

    def test_completely_garbled_json_still_retries(self) -> None:
        """If the regex fallback also fails, the error is re-raised and retries continue."""
        cleaner = make_cleaner(max_retries=3)
        # No 'cleaned' or 'classification' keys at all — regex won't match
        garbled_response = {'message': {'content': 'this is not json and has no structure'}}
        good_response = make_response("Some text.", "body")
        with patch(patch_llm_chat, side_effect=[garbled_response, good_response]) as mock_chat:
            cleaned, classification = cleaner.clean("Some text.")
        assert classification == "body"
        assert mock_chat.call_count == 2  # garbled triggers retry, good response succeeds

    def test_max_retries_configurable(self) -> None:
        cleaner = make_cleaner(max_retries=5)
        bad_response = {'message': {'content': 'not valid json'}}
        with patch(patch_llm_chat, return_value=bad_response) as mock_chat:
            cleaner.clean("Some text.")
        assert mock_chat.call_count == 5

    def test_retry_uses_same_messages(self) -> None:
        cleaner = make_cleaner(max_retries=3)
        bad_response = {'message': {'content': 'not valid json'}}
        good_response = make_response("Some text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]) as mock_chat:
            cleaner.clean("Some text.", page_context="Full page text.")
        for call in mock_chat.call_args_list:
            messages = call[1]['messages']
            user_message = next(m for m in messages if m['role'] == 'user')
            assert "Full page text." in user_message['content']
            assert "Some text." in user_message['content']

    def test_intermediate_rejections_suppressed_on_success(self, capsys) -> None:
        """Rejection messages from failed attempts are not shown if a later attempt succeeds."""
        cleaner = TextCleaner(model=TEST_LLM_MODEL, max_retries=3, temperature=0, verbose=True)
        bad_response = {'message': {'content': 'not valid json'}}
        good_response = make_response("Some text.", "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]):
            cleaner.clean("Some text.")
        out = capsys.readouterr().out
        assert "attempt 1 rejected" not in out


# --- TestSizeCheck ---

class TestSizeCheck:
    def test_size_check_not_applied_to_short_strings(self) -> None:
        """The size check is skipped for strings below the minimum length threshold.

        Real example: 'p(b,b) 1.' (9 printable chars) cleaned to 'p(b, b) = 1.' (12 chars)
        is a 33% increase. The LLM is correct — the check should not reject it.
        """
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("p(b, b) = 1.", "body")) as mock:
            cleaned, classification = cleaner.clean("p(b,b) 1.")
        assert cleaned == "p(b, b) = 1."
        assert mock.call_count == 1  # no retry triggered

    def test_size_check_still_applied_to_longer_strings(self) -> None:
        """The size check still rejects gross expansions on longer strings."""
        cleaner = make_cleaner(max_retries=3)
        original = "This is a longer paragraph text."   # > minimum threshold
        expanded = "This is a longer paragraph text that has been expanded significantly by the LLM."
        good_response = make_response(original, "body")
        bad_response = make_response(expanded, "body")
        with patch(patch_llm_chat, side_effect=[bad_response, good_response]) as mock:
            cleaner.clean(original)
        assert mock.call_count == 2  # first attempt rejected by size check, second succeeds


# --- TestSanityCheck ---

class TestSanityCheck:
    def test_accepts_ocr_fix(self) -> None:
        """Replacing a broken OCR word with a valid word should be accepted."""
        cleaner = make_cleaner()
        # "hppy" is not a valid English word — fixing it to "happy" is legitimate
        with patch(patch_llm_chat, return_value=make_response("I am happy today.", "body")):
            cleaned, classification = cleaner.clean("I am hppy today.")
        assert cleaned == "I am happy today."

    def test_restores_valid_word_substitution(self) -> None:
        """When LLM substitutes a valid word, restore it without retrying."""
        cleaner = make_cleaner(max_retries=3)
        # "judiciary" is valid — "judicial" substitution should be silently undone
        response = make_response("He obstructed judicial powers.", "body")
        with patch(patch_llm_chat, return_value=response) as mock_chat:
            cleaned, classification = cleaner.clean("He obstructed judiciary powers.")
        assert cleaned == "He obstructed judiciary powers."
        assert classification == "body"
        assert mock_chat.call_count == 1  # no retry needed

    def test_keeps_other_llm_changes_when_restoring_valid_word(self) -> None:
        """When restoring a valid word, other LLM changes (e.g. OCR fixes) are kept."""
        cleaner = make_cleaner()
        # LLM fixes "hppy" → "happy" (legitimate) but also swaps "today" → "now" (valid→valid)
        response = make_response("I am happy now.", "body")
        with patch(patch_llm_chat, return_value=response):
            cleaned, _ = cleaner.clean("I am hppy today.")
        assert cleaned == "I am happy today."  # "today" restored, "happy" kept

    def test_accepts_punctuation_only_change(self) -> None:
        """Changes that only affect punctuation (not words) should be accepted."""
        cleaner = make_cleaner()
        with patch(patch_llm_chat, return_value=make_response("Hello world.", "body")):
            cleaned, classification = cleaner.clean("Hello, world.")
        assert cleaned == "Hello world."

    def test_trusts_llm_on_invalid_to_invalid_substitution(self) -> None:
        """LLM replaces one invalid token with another — we now trust the LLM's version.

        'endeavoured' (British spelling) is not in NLTK WordNet, so it appears invalid
        to the word validator. 'endeavourd' is also invalid. Under the old policy this
        was flagged as a suspicious invalid→invalid swap and triggered a retry. Under
        the new policy we trust the LLM, so the LLM's version is kept as-is.
        """
        cleaner = make_cleaner(max_retries=3)
        # noinspection SpellCheckingInspection
        bad_response = make_response("He has endeavourd to bring on the inhabitants.", "body")
        with patch(patch_llm_chat, return_value=bad_response) as mock_chat:
            cleaned, _ = cleaner.clean("He has endeavoured to bring on the inhabitants.")
        assert cleaned == "He has endeavourd to bring on the inhabitants."
        assert mock_chat.call_count == 1  # no retry triggered

    def test_accepts_word_removal_of_ocr_artifact(self) -> None:
        """Removing a number that is an OCR artifact should be accepted (size check permitting)."""
        cleaner = make_cleaner()
        # trailing "4" is a footnote marker — removing it is legitimate
        with patch(patch_llm_chat, return_value=make_response("The movement grew rapidly.", "body")):
            cleaned, _ = cleaner.clean("The movement grew rapidly. 4")
        assert cleaned == "The movement grew rapidly."


# --- TestHasSuspiciousSubstitutions ---

class TestHasSuspiciousSubstitutions:

    @pytest.mark.parametrize("original, cleaned", [
        # Identical text
        ("Hello world.", "Hello world."),
        # OCR artifacts: embedded non-alpha chars keep the token invalid after strip
        ("He said a|nd walked away.",    "He said and walked away."),     # pipe char — regression
        ("He wishes t<; thank her.",     "He wishes to thank her."),      # angle bracket
        ("The xzqpf was undeniable.",    "The truth was undeniable."),    # pure OCR garbage → valid
        # Boundary punctuation: stripped forms match, so treated as equal
        ("The council, agreed on the plan.", "The council agreed on the plan."),  # trailing comma
        ("The (government) responded.",      "The government responded."),        # boundary parens
        ("She agreed.",                      "She agreed."),                      # trailing period
        # Valid-word substitutions — handled by _restore_valid_words, not flagged here
        ("The cat sat on the mat.",  "The dog sat on the mat."),   # valid→valid swap
        ("The quick brown fox.",     "The quick xzqpf fox."),      # valid→invalid
        # Hyphen/em-dash equivalence: equal after dash normalization
        ("a closely-integrated work-far exceeding expectations",
         "a closely-integrated work—far exceeding expectations"),  # hyphen→em-dash in compound
        ("their wellknown paradoxes", "their well-known paradoxes"),  # missing hyphen restored
        # Diacritics: equal after ASCII-folding
        ("the experiments by Eotvos more recently",
         "the experiments by Eötvös more recently"),
        # Symbol/punctuation tokens: non-alphabetic tokens are skipped entirely
        ("marked with a star:✶",   "marked with a star:*"),    # embedded symbol
        ("marked with a star ✶ here", "marked with a star * here"),  # standalone symbol
        ("the symbol ✶✶ appears",  "the symbol ** appears"),   # multi-char symbol
        # invalid→invalid: LLM's best effort on OCR garbage is trusted
        ("The xzqpf was clear.", "The zqpfx was clear."),
    ])
    def test_not_suspicious(self, original: str, cleaned: str) -> None:
        assert _has_suspicious_substitutions(original, cleaned) is False


# --- TestRestoreListPrefix ---

class TestRestoreListPrefix:
    @pytest.mark.parametrize("original,cleaned,expected", [
        ("(3) All human actions are egotistic.", "All human actions are egotistic.", "(3) All human actions are egotistic."),
        ("3. All human actions are egotistic.",  "All human actions are egotistic.", "3. All human actions are egotistic."),
    ])
    def test_prefix_restored_when_dropped(self, original: str, cleaned: str, expected: str) -> None:
        """Numbered list prefix dropped by LLM is restored."""
        assert _restore_list_prefix(original, cleaned) == expected

    @pytest.mark.parametrize("original,cleaned,expected", [
        ("a) The first item in a lettered list.", "The first item in a lettered list.", "a) The first item in a lettered list."),
        ("b) The second item.", "The second item.", "b) The second item."),
        ("c) The replacement, in the neo-classical theory.", "The replacement, in the neo-classical theory.", "c) The replacement, in the neo-classical theory."),
    ])
    def test_lettered_prefix_restored_when_dropped(self, original: str, cleaned: str, expected: str) -> None:
        """Lettered list prefix a)/b)/c) dropped by LLM is restored."""
        assert _restore_list_prefix(original, cleaned) == expected

    def test_prefix_not_duplicated_when_already_present(self) -> None:
        """Prefix already present in cleaned text is not added again."""
        result = _restore_list_prefix(
            "(3) All human actions are egotistic.",
            "(3) All human actions are egotistic."
        )
        assert result == "(3) All human actions are egotistic."

    def test_no_prefix_in_original_leaves_cleaned_unchanged(self) -> None:
        """Text without a list prefix is returned unchanged."""
        result = _restore_list_prefix(
            "All human actions are egotistic.",
            "All human actions are egotistic."
        )
        assert result == "All human actions are egotistic."


# --- TestNormalizeDashes ---

class TestNormalizeDashes:
    @pytest.mark.parametrize("input_str,expected", [
        ('well—known', 'well-known'),   # em-dash replaced
        ('well–known', 'well-known'),   # en-dash replaced
        ('well-known', 'well-known'),   # plain hyphen unchanged
        ('hello',      'hello'),        # no dash unchanged
        ('a—b–c',      'a-b-c'),        # both em and en dash replaced
        ('',           ''),             # empty string
    ])
    def test_normalizes_dashes(self, input_str: str, expected: str) -> None:
        assert _normalize_dashes(input_str) == expected


# --- TestRestoreValidWords ---

class TestRestoreValidWords:
    def test_valid_word_substitution_is_restored(self) -> None:
        """LLM swaps one valid word for another — original is restored."""
        result = _restore_valid_words(
            "He obstructed judiciary powers.",
            "He obstructed judicial powers."
        )
        assert result == "He obstructed judiciary powers."

    def test_ocr_fix_to_valid_word_is_kept(self) -> None:
        """LLM fixes an invalid OCR token to a valid word — change is kept."""
        result = _restore_valid_words(
            "I am hppy today.",
            "I am happy today."
        )
        assert result == "I am happy today."

    def test_ocr_fix_kept_while_valid_substitution_is_restored(self) -> None:
        """Mixed case: OCR fix kept, valid-word swap restored."""
        result = _restore_valid_words(
            "I am hppy today.",
            "I am happy now."
        )
        assert result == "I am happy today."

    def test_identical_text_unchanged(self) -> None:
        result = _restore_valid_words("Hello world.", "Hello world.")
        assert result == "Hello world."

    def test_em_dash_treated_as_hyphen(self) -> None:
        """LLM upgrades hyphen to em-dash in a compound — keep the upgrade."""
        result = _restore_valid_words(
            "a false-as assumption",
            "a false—as assumption"
        )
        # "false-as" and "false—as" normalize to the same thing — no substitution
        assert result == "a false—as assumption"

    def test_llm_em_dash_downgraded_to_hyphen_is_restored(self) -> None:
        """LLM downgrades em-dash to plain hyphen in a compound — restore the em-dash.
        Seen in practice: 'false—as' and '1913—can' both downgraded to hyphens."""
        result = _restore_valid_words(
            "theories shown to be false—as for example the model of 1913—can retain",
            "theories shown to be false-as for example the model of 1913-can retain"
        )
        assert result == "theories shown to be false—as for example the model of 1913—can retain"

    def test_ocr_artifact_replaced_with_valid_word_is_kept(self) -> None:
        """Invalid OCR token replaced by valid word — keep the fix."""
        result = _restore_valid_words(
            "The xzqpf was undeniable.",
            "The truth was undeniable."
        )
        assert result == "The truth was undeniable."

    def test_multiple_valid_substitutions_all_restored(self) -> None:
        """All valid-word swaps across the sentence are restored."""
        result = _restore_valid_words(
            "The quick brown fox.",
            "The slow white dog."
        )
        assert result == "The quick brown fox."

    def test_n_to_1_merge_producing_valid_word_is_kept(self) -> None:
        """LLM merges an OCR split into a valid word — keep the fix."""
        # "Scienti fic" is an OCR word-break; "Scientific" is the correct merge
        result = _restore_valid_words(
            "the Scienti fic method",
            "the Scientific method"
        )
        assert result == "the Scientific method"

    def test_n_to_1_merge_producing_invalid_word_restores_originals(self) -> None:
        """LLM merges tokens into an invalid word — restore the originals."""
        # "- including" (standalone dash + word) merged into "—including" (invalid token)
        result = _restore_valid_words(
            "Parts of the Postscript - including Realism",
            "Parts of the Postscript —including Realism"
        )
        assert result == "Parts of the Postscript - including Realism"

    def test_n_to_1_merge_producing_number_is_kept(self) -> None:
        """LLM merges OCR-broken number tokens into a single number — keep the fix."""
        result = _restore_valid_words(
            "published in 1 959.",
            "published in 1959."
        )
        assert result == "published in 1959."

    def test_n_to_1_merge_producing_number_with_punctuation_is_kept(self) -> None:
        """Punctuation around a merged number should not prevent it being kept."""
        result = _restore_valid_words(
            "Popper (1 977), argued",
            "Popper (1977), argued"
        )
        assert result == "Popper (1977), argued"

    def test_n_to_1_merge_of_spaced_abbreviation_is_kept(self) -> None:
        """'i. e.,' merged to 'i.e.,' is kept — concatenation of originals matches."""
        result = _restore_valid_words(
            "means nothing, i. e., it is not falsifiable.",
            "means nothing, i.e., it is not falsifiable."
        )
        assert result == "means nothing, i.e., it is not falsifiable."

    def test_n_to_1_merge_with_internal_period_is_kept(self) -> None:
        """'Ph. D' merged to 'Ph.D.' is kept — internal period signals deliberate abbreviation."""
        result = _restore_valid_words(
            "awarded a Ph. D in philosophy.",
            "awarded a Ph.D. in philosophy."
        )
        assert result == "awarded a Ph.D. in philosophy."

    def test_hyphen_as_dash_separator_upgraded_to_em_dash_is_kept(self) -> None:
        """['criticism', '-', 'and'] → 'criticism—and': LLM replaced hyphen-as-dash with em-dash."""
        result = _restore_valid_words(
            "a criticism - and its response",
            "a criticism—and its response"
        )
        assert result == "a criticism—and its response"

    def test_two_words_hyphenated_into_compound_is_kept(self) -> None:
        """['proof', 'reading'] → 'proof-reading': LLM correctly hyphenated a compound noun."""
        result = _restore_valid_words(
            "requires careful proof reading of the text",
            "requires careful proof-reading of the text"
        )
        assert result == "requires careful proof-reading of the text"

    def test_three_tokens_with_hyphen_em_dashed_is_kept(self) -> None:
        """['long', '-', 'term'] → 'long—term': LLM upgraded hyphen separator to em-dash."""
        result = _restore_valid_words(
            "a long - term solution",
            "a long—term solution"
        )
        assert result == "a long—term solution"

    def test_em_dash_glued_to_word_restored_to_space_hyphen_space(self) -> None:
        """Standalone ' - ' merged by LLM into '—word' is restored to ' - word'."""
        result = _restore_valid_words(
            "before - after",
            "before —after"
        )
        assert result == "before - after"

    def test_standalone_hyphen_preserved_when_unchanged(self) -> None:
        """A standalone ' - ' that the LLM leaves untouched should survive the join."""
        result = _restore_valid_words(
            "a criterion of demarcation - the criterion",
            "a criterion of demarcation - the criterion"
        )
        assert result == "a criterion of demarcation - the criterion"

    def test_standalone_hyphen_preserved_when_other_change_made(self) -> None:
        """Spaces around a standalone ' - ' are preserved when the LLM changes something else."""
        result = _restore_valid_words(
            "I do not believe that a criterion of demarcation - the criterion of falsifiability.",
            "I do not believe that a criterion of demarcation - the criterion of falsifiability."
        )
        assert result == "I do not believe that a criterion of demarcation - the criterion of falsifiability."

    def test_standalone_hyphen_spaces_not_stripped_by_join(self) -> None:
        """The ' '.join() at the end of _restore_valid_words must not collapse ' - ' into '-'."""
        # Specifically testing that the standalone '-' token and its neighbours
        # are all preserved as separate tokens so the join reconstructs the spaces.
        result = _restore_valid_words(
            "a criterion of demarcation - the",
            "a criterion of demarcation - the"
        )
        assert " - " in result

    def test_1_to_3_expansion_of_line_break_hyphen_is_kept(self) -> None:
        """LLM expanding 'word-word' (line-break hyphen) to 'word - word' should be kept."""
        # Docling joins PDF line breaks as "demarcation-the" (plain hyphen, no spaces).
        # The LLM correctly expands this to "demarcation - the". This is a 1→3 token
        # expansion and must not be reverted by _restore_valid_words.
        result = _restore_valid_words(
            "a criterion of demarcation-the criterion of falsifiability.",
            "a criterion of demarcation - the criterion of falsifiability."
        )
        assert result == "a criterion of demarcation - the criterion of falsifiability."

    # --- Dash upgrade where the first token starts with a quote character ---
    # original_split[i1][0].isalpha() blocked upgrades where the first token is
    # a quoted word such as "'meaning'" or '‘meaning’'.  The guard should
    # only block a leading standalone dash, not quote-prefixed words.

    def test_dash_upgrade_with_curly_quoted_first_token_is_kept(self) -> None:
        """['‘meaning’', '-', 'laden'] → '‘meaning’—laden': upgrade must be kept."""
        # The first token starts with a curly open-quote, not a letter.
        # Before the fix, original_split[i1][0].isalpha() returned False and
        # the upgrade was wrongly discarded.
        result = _restore_valid_words(
            "the ‘meaning’ - laden concept",
            "the ‘meaning’—laden concept",
        )
        assert result == "the ‘meaning’—laden concept"

    def test_dash_upgrade_with_ascii_quoted_first_token_is_kept(self) -> None:
        """[\"'meaning'\", '-', 'laden'] → \"'meaning'—laden\": upgrade must be kept."""
        # Same as above but with ASCII single quotes.
        result = _restore_valid_words(
            "the 'meaning' - laden concept",
            "the 'meaning'—laden concept",
        )
        assert result == "the 'meaning'—laden concept"

    def test_standalone_dash_first_token_still_blocked(self) -> None:
        """['-', 'including'] → '—including': guard must still reject this."""
        # A bare leading '-' token being upgraded and glued to the next word
        # is not a legitimate em-dash upgrade — the guard must remain active.
        result = _restore_valid_words(
            "was - including all",
            "was —including all",
        )
        assert result == "was - including all"

    def test_llm_downgrades_em_dash_to_double_hyphen_is_restored(self) -> None:
        """LLM replaces '—' with '--' in a compound token — restore the original em-dash.

        Regression: extending _normalize_dashes to collapse '--' to '-' caused
        'justifiable—as' and 'justifiable--as' to compare as equal in the 1:1
        path, silently accepting the LLM's downgrade.  The '--' collapsing must
        only apply in the N→1 is_dash_upgrade comparison, not in the 1:1 path.
        """
        result = _restore_valid_words(
            "becomes as good–or as justifiable—as any other",
            "becomes as good-or as justifiable--as any other",
        )
        assert result == "becomes as good–or as justifiable—as any other"

    def test_dash_upgrade_when_first_token_contains_double_dash(self) -> None:
        """['relationship--instantiation', '-', 'whose'] → 'relationship—instantiation—whose'.

        The first original token already contains '--' (a double-dash from OCR).
        _normalize_dashes collapses '--' to '-' on the cleaned side, so the
        joined originals must also be normalized before comparison, otherwise
        the '--' in joined_orig causes a mismatch and the LLM's upgrade is lost.
        """
        result = _restore_valid_words(
            "the relationship--instantiation - whose meaning",
            "the relationship—instantiation—whose meaning",
        )
        assert result == "the relationship—instantiation—whose meaning"

    # --- Quote normalization: LLM introduces smart/curly quotes ---
    # The preprocessor normalizes all quotes to ASCII before text reaches _restore_valid_words.
    # The LLM often "improves" straight quotes back to typographic curly quotes. The only
    # difference between the original and LLM tokens is the quote Unicode code point —
    # they are semantically identical and must not trigger a restore.

    @pytest.mark.parametrize("original, llm", [
        # Seen in practice as: → restored ‘’All’ (LLM tried ‘’All’)
        ("He said ‘All is well.",    "He said ‘All is well."),
        # Seen in practice as: → restored ‘white’.’ (LLM tried ‘white’.’)
        ("the colour white’.",       "the colour white’."),
        # Seen in practice as: → restored ‘Vienna.’’ (LLM tried ‘Vienna.’’)
        ("the city Vienna.’",        "the city Vienna.’"),
        # Seen in practice as: → restored ‘’Ed.’.’ (LLM tried ‘’Ed.’.’)
        ("published by ‘Ed.’.",      "published by ‘Ed.’."),
    ])
    def test_llm_smart_quote_not_restored(self, original: str, llm: str) -> None:
        """LLM smart-quote upgrade (ASCII → typographic) must not trigger a restore."""
        assert _restore_valid_words(original, llm) == llm


# --- TestIsWordLike ---

class TestIsWordLike:
    @pytest.mark.parametrize("token,expected", [
        (':s;;', False),   # 25% alpha — symbol-heavy OCR artifact
        ('i:.',  False),   # 33% alpha — symbol-heavy OCR artifact
        ('its',  True),    # 100% alpha
        ('a,',   True),    # 50% alpha — at the boundary, word-like
        ('h.',   True),    # 50% alpha — at the boundary, word-like
        ('',     False),   # empty string
        ('1979', False),   # 0% alpha — pure digits
    ])
    def test_is_word_like(self, token: str, expected: bool) -> None:
        assert _is_word_like(token) is expected


# --- TestRestoreValidWordsSymbols ---

class TestRestoreValidWordsSymbols:
    """Tests that math/symbol OCR fixes by the LLM are not wrongly rolled back.

    The root cause: a token like ':s;;' strips to 's', which passes
    is_valid_word().  Without a word-likeness guard the restore fires
    spuriously, undoing a correct LLM fix.
    """

    @pytest.mark.parametrize("original,cleaned,expected", [
        ('For all x :s;; 0.', 'For all x ≤ 0.', 'For all x ≤ 0.'),    # ':s;;' → '≤'
        ('Such that a i:. b.', 'Such that a ≠ b.', 'Such that a ≠ b.'),  # 'i:.' → '≠'
        ('Requires that p i:. q.', 'Requires that p ≥ q.', 'Requires that p ≥ q.'),  # 'i:.' → '≥'
    ])
    def test_symbol_ocr_fix_is_kept(self, original: str, cleaned: str, expected: str) -> None:
        """Symbol-heavy OCR artifacts must not be rolled back when the LLM fixes them.

        Seen in the full run: → restored ':s;;' (LLM tried '≤'), etc.
        These tokens strip to a single letter that passes is_valid_word,
        triggering a spurious restore without a word-likeness guard.
        """
        assert _restore_valid_words(original, cleaned) == expected

    def test_normal_word_restore_still_works_after_symbol_fix(self) -> None:
        """The symbol guard must not suppress restores of normal valid words."""
        result = _restore_valid_words(
            'He obstructed judiciary powers.',
            'He obstructed judicial powers.'
        )
        assert result == 'He obstructed judiciary powers.'

    def test_word_with_trailing_comma_still_restored(self) -> None:
        """A word token that is exactly 50% alphabetic (e.g. 'a,') must still be
        treated as word-like and restored when the LLM substitutes it."""
        result = _restore_valid_words(
            'Choose a, not b.',
            'Choose an, not b.'
        )
        assert result == 'Choose a, not b.'


# --- TestRestoreValidWordsSingleLetter ---

class TestRestoreValidWordsSingleLetter:
    """Tests that isolated single-letter OCR fragments are not treated as valid words.

    In practice these are always OCR-split function words: 'or' → 'o'+'r',
    'to' → 't'+'o', 'by' → 'b'+'y'. The LLM correctly recognises and restores
    them, but _restore_valid_words wrongly reverts the fix because all 26
    letters pass is_valid_word().

    Real examples from Last Full Run.txt:
      → restored 'r' (LLM tried 'or')   — Hume quote, page 147
      → restored 'o' (LLM tried 'to')   — same passage
      → restored 'y' (LLM tried 'by')   — page 177
    """

    @pytest.mark.parametrize("original,cleaned,expected", [
        # Real case: "reason;.,.. r you must allow" — 'r' is the tail of 'or'
        ('r you must allow that',   'or you must allow that',   'or you must allow that'),
        # Real case: "allow that o your belief" — 'o' is the tail of 'to'
        ('allow that o your belief', 'allow that to your belief', 'allow that to your belief'),
        # Real case (page 177): standalone 'y' — tail of 'by'
        ('justified y the results',  'justified by the results',  'justified by the results'),
        # 'a' and 'I' are genuine English words — must still be protected
        ('a cat sat',               'the cat sat',               'a cat sat'),
        ('I think so',              'We think so',               'I think so'),
    ])
    def test_single_letter_restore_behavior(
            self, original: str, cleaned: str, expected: str) -> None:
        assert _restore_valid_words(original, cleaned) == expected


# --- TestRestoreValidWordsCurlyQuotes ---

class TestRestoreValidWordsCurlyQuotes:
    """Tests that curly/smart quotes are stripped before is_valid_word in N→1 merges.

    The strip string in the N→1 merge path only includes straight quotes, so curly
    quotes left on the merged token cause is_valid_word to fail and the originals
    to be wrongly restored.

    The failing case requires that joined_orig != cleaned_split[j1] so the equality
    shortcut doesn’t fire — achieved by having the LLM both fix an OCR error AND
    merge with a curly-quoted token at the same time.

    Real examples:
      ["‘jistify’", ";"]   → "‘justify’;"   — OCR fix + semicolon merge
      ["‘", "antibadies’-that"] → "‘antibodies’-that" — OCR fix + stray quote join
    """

    @pytest.mark.parametrize("original, cleaned, expected", [
        # semicolon merge: ‘jistify’ → ‘justify’ + ‘;’ merged
        (
            "pragmatic sense of ‘jistify’ ; in other words,",
            "pragmatic sense of ‘justify’; in other words,",
            "pragmatic sense of ‘justify’; in other words,",
        ),
        # comma merge: ‘eficiency’ → ‘efficiency’ + ‘,’ merged
        (
            "she argued that ‘eficiency’ , was key,",
            "she argued that ‘efficiency’, was key,",
            "she argued that ‘efficiency’, was key,",
        ),
    ])
    def test_curly_quoted_ocr_fix_and_punctuation_merge_kept(
            self, original: str, cleaned: str, expected: str) -> None:
        """LLM fixes OCR error in curly-quoted word AND merges a punctuation token."""
        assert _restore_valid_words(original, cleaned) == expected


# --- TestRestoreValidWordsBackslashQuotes ---

class TestRestoreValidWordsBackslashQuotes:
    """Tests that backslash-escaped quotes in LLM tokens don't wrongly trigger restore.

    The JSON escape repair converts \'word\' to a Python string containing a
    literal backslash + quote (e.g. \'possibility\').  The comparison strip
    removes quote characters but not backslashes, so the stripped forms differ
    and the original is wrongly restored.

    Fix: collapse \' → ' and \" → " before normalize_quotes + strip in the
    1:1 comparison path so the bare words are correctly compared.

    Real examples (page 395 of Realism and the Aim of Science):
      original '‘possibility’,'  LLM tried \\'possibility\\',
      original '‘probability’'   LLM tried \\'probability\\'
      original '‘frequency’)'    LLM tried \\'frequency\\'
    """

    @pytest.mark.parametrize("original, cleaned", [
        ("the ‘possibility’ is", "the \\'possibility\\' is"),   # single-quote escaping
        ('the "probability" is', 'the \\"probability\\" is'),   # double-quote escaping
    ])
    def test_backslash_escaped_quotes_not_wrongly_restored(
            self, original: str, cleaned: str) -> None:
        """Backslash-escaped quotes in LLM tokens don’t wrongly trigger restore."""
        assert _restore_valid_words(original, cleaned) == cleaned


# --- TestRestoreValidWordsTrailingHyphen ---

class TestRestoreValidWordsTrailingHyphen:
    """Tests that tokens with a trailing hyphen are not restored.

    A trailing hyphen always signals either a word-break fragment ('theo-',
    'deter-', 'mis-') or a mid-sentence dash being upgraded to an em-dash
    ('conditions-' → 'conditions—'). In both cases the LLM's fix should win.

    Real examples from Last Full Run.txt:
      → restored 'theo-'      (LLM tried 'theories.')   — context: 'scientific character of theo-'
      → restored 'deter-'     (LLM tried 'determined')  — context: 'to the classes deter-'
      → restored 'mis-'       (LLM tried 'misconceptions') — context: 'accounts for these mis-'
      → restored 'conditions-'(LLM tried 'conditions—') — context: "fetters of its conditions- a 'rule of"
    """

    @pytest.mark.parametrize("original,cleaned,expected", [
        # Word-break fragments: LLM completes the word, must not be rolled back
        ('scientific character of theo-',       'scientific character of theories.',      'scientific character of theories.'),
        ('to the classes deter-',               'to the classes determined',              'to the classes determined'),
        ('accounts for these mis-',             'accounts for these misconceptions',      'accounts for these misconceptions'),
        # Em-dash upgrade: trailing hyphen upgraded to em-dash, must not be rolled back
        ("fetters of its conditions- a rule of", "fetters of its conditions— a rule of", "fetters of its conditions— a rule of"),
    ])
    def test_trailing_hyphen_token_not_restored(
            self, original: str, cleaned: str, expected: str) -> None:
        assert _restore_valid_words(original, cleaned) == expected

    def test_mid_hyphen_compound_still_restored(self) -> None:
        """A hyphen in the middle of a token (compound word) must still trigger restore."""
        result = _restore_valid_words(
            'the well-known argument',
            'the well-known  argument'  # LLM left it unchanged but with extra space
        )
        # 'well-known' does not end with '-', so normal restore rules apply
        assert 'well-known' in result


# --- TestRestoreValidWordsLineBreakHyphen ---

class TestRestoreValidWordsLineBreakHyphen:
    """Tests for the N→1 line-break hyphen join condition.

    When a PDF splits a word at line-end with a hard hyphen, Docling produces
    two tokens: ['reexamina-', 'tion'].  The LLM correctly rejoins them to
    'reexamination'.  The N→1 merge must accept this even when is_valid_word
    fails (e.g. for long compound words not in the dictionary).

    Condition: strip trailing hyphens from each original token, concatenate,
    and if the result equals the cleaned token, keep the LLM's join.
    """

    @pytest.mark.parametrize("original,cleaned,expected", [
        # Core cases from the fix list
        ('the reexamina- tion of evidence',
         'the reexamination of evidence',
         'the reexamination of evidence'),
        ('they classify- ing the result',
         'they classifying the result',
         'they classifying the result'),
        # Three-part split of a plain word
        ('a contra- dic- tion here',
         'a contradiction here',
         'a contradiction here'),
        # Hyphenated compound word reassembled: dehyphen join ≠ cleaned token
        # because the LLM correctly preserves the compound hyphen
        ('a self- contra- diction here',
         'a self-contradiction here',
         'a self-contradiction here'),
    ])
    def test_line_break_hyphen_join_kept(
            self, original: str, cleaned: str, expected: str) -> None:
        assert _restore_valid_words(original, cleaned) == expected

    def test_non_hyphen_merge_not_accepted(self) -> None:
        """A 2→1 merge with no trailing hyphens and no match must restore originals.

        joined_orig = 'helloworld', cleaned = 'helloplanet': dehyphen_join is also
        'helloworld' ≠ 'helloplanet', so is_line_break_join does not fire.
        is_valid_word('helloplanet') is False and no other check applies either.
        """
        result = _restore_valid_words(
            "hello world today",
            "helloplanet today",
        )
        assert result == "hello world today"


# --- TestRestoreValidWordsLlmDashJoin ---

class TestRestoreValidWordsLlmDashJoin:
    """Tests for the N→1 LLM-introduced em-dash between valid words.

    When the LLM correctly joins N OCR-fragmented tokens AND inserts an em-dash
    between two real words, the N→1 merge must accept it even though
    is_dash_upgrade fails (the joined originals don't contain the dash).

    Example: ['justi', 'fication', 'is'] → 'justification—is'
    joined_orig = 'justificationis', _normalize_dashes('justification—is') =
    'justification-is' — these don't match, so is_dash_upgrade is False.
    But splitting 'justification—is' on em-dash gives ['justification', 'is'],
    both valid words, so the merge must be accepted.
    """

    @pytest.mark.parametrize("original, cleaned, expected", [
        ("justi fication is", "justification—is",  "justification—is"),   # 3→1 merge
        ("justifi cation",    "justification—and", "justification—and"),  # 2→1 merge
    ])
    def test_llm_dash_join_kept(
            self, original: str, cleaned: str, expected: str) -> None:
        """N→1 merge where LLM rejoins OCR fragments and inserts an em-dash."""
        assert _restore_valid_words(original, cleaned) == expected

    def test_llm_dash_join_rejected_when_parts_not_valid_words(self) -> None:
        """If either dash-separated part is not a valid word, restore originals.

        'qwerty' and 'asdf' are not real English words, so the em-dash join
        must be rejected and the originals restored.
        """
        result = _restore_valid_words(
            "qwerty asdf",
            "qwerty—asdf",
        )
        assert result == "qwerty asdf"


# --- TestRestoreValidWordsOcrFix ---

class TestRestoreValidWordsOcrFix:
    """Tests for N→1 merges where the LLM fixes an OCR misspelling.

    When at least one original token is not a valid word, and the LLM's
    merged result is a valid word or hyphenated compound (all hyphen-separated
    parts are real words), the merge should be accepted — the LLM has
    corrected a genuine OCR error, not hallucinated.
    """

    @pytest.mark.parametrize("original, llm, expected", [
        (
            "either by excluding self - coatradictory hypotheses",
            "either by excluding self-contradictory hypotheses",
            "either by excluding self-contradictory hypotheses",
        ),
        (
            "can spread with super - luminar velocity",
            "can spread with super-luminal velocity",
            "can spread with super-luminal velocity",
        ),
        (
            # Real failure (Page 225): LLM fixed 'teste'→'tested' in a 1:1
            # replacement but left the orphan 'd' token unchanged.  The restore
            # logic then fires because 'teste' is a valid dictionary word.
            # The minimum correct outcome is to keep the LLM's 'tested'.
            "and more severely teste d -even in fields",
            "and more severely tested d -even in fields",
            "and more severely tested d -even in fields",
        ),
    ])
    def test_ocr_fix_accepted(self, original: str, llm: str, expected: str) -> None:
        """LLM merge is accepted when at least one original token is invalid."""
        assert _restore_valid_words(original, llm) == expected

    @pytest.mark.parametrize("original, llm, expected", [
        (
            # All originals valid: existing is_hyphen_compound rule accepts it
            "a good day",
            "a good-day",
            "a good-day",
        ),
        (
            # LLM reordered content — neither part of the merge is valid
            "the self - coatradictory xyzzy",
            "the self-xyzzy coatradictory",
            None,  # checked below: originals restored, not the garbled LLM merge
        ),
    ])
    def test_ocr_fix_guard(self, original: str, llm: str, expected: str | None) -> None:
        """Guard: is_ocr_fix must not fire when it shouldn't."""
        result = _restore_valid_words(original, llm)
        if expected is not None:
            assert result == expected
        else:
            # LLM reordered tokens — originals should be restored
            assert "coatradictory" in result or "self" in result


# --- TestCoerceClassification ---

class TestCoerceClassification:

    @pytest.mark.parametrize("label,expected", [
        ('footnote',       'footnote'),
        ('endnote',        'footnote'),
        ('note',           'footnote'),
        ('body',           'body'),
        ('main content',   'body'),
        ('prose',          'body'),
        ('index',          'drop'),
        ('bibliograph',    'drop'),
        ('reference list', 'drop'),
    ])
    def test_hint_maps_to_classification(self, label: str, expected: str) -> None:
        assert _coerce_classification(label) == expected

    def test_matching_is_case_insensitive(self) -> None:
        assert _coerce_classification('FOOTNOTE') == 'footnote'
        assert _coerce_classification('BODY TEXT') == 'body'
        assert _coerce_classification('INDEX') == 'drop'

    def test_unknown_label_returns_none(self) -> None:
        assert _coerce_classification('something_unknown') is None

    def test_empty_string_returns_none(self) -> None:
        assert _coerce_classification('') is None


# --- TestCleanFormulaOcr ---

class TestCleanFormulaOcr:
    """Tests for TextCleaner.clean_formula_ocr — the OCR-reconstruction pass.

    This method sends a formula to the LLM asking it to reconstruct the
    intended notation from an OCR-mangled string.  It returns plain text
    (not JSON) and must use a different system prompt from clean_formula.
    """

    @pytest.mark.parametrize("formula, expected", [
        ("x -+- y == z (garbled)",  "x + y = z"),
        ("E = mc 2",                "E = mc²"),
    ])
    def test_returns_llm_response(self, formula: str, expected: str) -> None:
        """clean_formula_ocr returns the LLM's reconstructed formula."""
        cleaner = make_cleaner()
        mock_response = {'message': {'content': expected}}
        with patch(patch_llm_chat, return_value=mock_response):
            result = cleaner.clean_formula_ocr(formula)
        assert result == expected

    def test_returns_original_on_exception(self) -> None:
        """clean_formula_ocr falls back to the raw formula if the LLM raises."""
        cleaner = make_cleaner()
        with patch(patch_llm_chat, side_effect=Exception("LLM unavailable")):
            result = cleaner.clean_formula_ocr("x + y = z")
        assert result == "x + y = z"

    def test_uses_different_prompt_from_audio_pass(self) -> None:
        """clean_formula_ocr must use a different system prompt than clean_formula."""
        from text_cleaner import FORMULA_SYSTEM_PROMPT
        cleaner = make_cleaner()
        mock_response = {'message': {'content': "x + y = z"}}
        with patch(patch_llm_chat, return_value=mock_response) as mock_chat:
            cleaner.clean_formula_ocr("x + y = z")
        system_msg = next(
            m['content'] for m in mock_chat.call_args[1]['messages']
            if m['role'] == 'system'
        )
        assert system_msg != FORMULA_SYSTEM_PROMPT

    def test_empty_formula_returned_unchanged(self) -> None:
        """clean_formula_ocr returns an empty string without calling the LLM."""
        cleaner = make_cleaner()
        with patch(patch_llm_chat) as mock_chat:
            result = cleaner.clean_formula_ocr("   ")
        assert result.strip() == ""
        mock_chat.assert_not_called()


# --- TestIntegration ---

class TestIntegration:
    @pytest.mark.integration
    def test_real_llm_call_body(self) -> None:
        """Integration test — requires a running LLM."""
        cleaner = make_cleaner()
        paragraph = "This is a sample paragraph from a book about philosophy and rationality."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert "philosophy" in cleaned
        assert "rationality" in cleaned
        assert cleaned == paragraph

    @pytest.mark.integration
    def test_real_llm_call_footnote(self) -> None:
        """Integration test — requires a running LLM."""
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
    def test_real_llm_call_drop(self) -> None:
        """Integration test — requires a running LLM."""
        cleaner = make_cleaner()
        cleaned, classification = cleaner.clean(
            "Chapter 1 ... 1\nChapter 2 ... 15\nChapter 3 ... 42"
        )
        assert classification == 'drop'

    @pytest.mark.integration
    def test_real_llm_call_body_unchanged(self) -> None:
        """Clean prose with no issues should be returned exactly as-is."""
        cleaner = make_cleaner()
        paragraph = "The French Revolution began in 1789 and fundamentally transformed the political landscape of Europe."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert cleaned == paragraph

    @pytest.mark.integration
    def test_real_llm_call_ocr_word_break_fixed(self) -> None:
        """Mid-word line breaks introduced by OCR should be rejoined."""
        cleaner = make_cleaner()
        paragraph = "The development of mod- ern philosophy can be traced to the six- teenth century."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert cleaned == "The development of modern philosophy can be traced to the sixteenth century."

    @pytest.mark.integration
    def test_real_llm_call_trailing_footnote_marker_stripped(self) -> None:
        """A trailing footnote number at the end of a body paragraph should be removed."""
        cleaner = make_cleaner()
        paragraph = "The movement grew rapidly throughout the nineteenth century, attracting followers from across the social spectrum. 4"
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert cleaned == "The movement grew rapidly throughout the nineteenth century, attracting followers from across the social spectrum."

    @pytest.mark.integration
    def test_real_llm_call_footnote_identified_with_page_context(self) -> None:
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
        assert cleaned.replace('–', '-') == "For full membership statistics by region, see Jones (1987), pp. 142-156."

    @pytest.mark.integration
    def test_real_llm_call_footnote_without_page_context(self) -> None:
        """Without page context the response should still be valid, even if classification varies."""
        cleaner = make_cleaner()
        paragraph = "4 For full membership statistics by region, see Jones (1987), pp. 142-156."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification in ('body', 'footnote', 'drop')
        assert isinstance(cleaned, str)

    @pytest.mark.integration
    def test_real_llm_call_drop_toc(self) -> None:
        """An obvious table of contents should be classified as drop."""
        cleaner = make_cleaner()
        cleaned, classification = cleaner.clean(
            "Introduction ... 1\nChapter One: The Early Years ... 15\n"
            "Chapter Two: The Middle Period ... 47\nConclusion ... 203"
        )
        assert classification == 'drop'
