"""Tests for the TextCleaner class."""

import pytest
from unittest.mock import patch
from text_cleaner import (TextCleaner, _has_suspicious_substitutions, _coerce_classification,
                          _normalize_dashes, _restore_valid_words, _restore_list_prefix)

patch_llm_chat: str = 'text_cleaner.ollama.chat'


# --- Fixtures ---

def make_cleaner(model: str = 'llama3.1:8b', max_retries: int = 3) -> TextCleaner:
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
        with patch(patch_llm_chat, return_value=make_response("A footnote.", "footnote")):
            cleaned, classification = cleaner.clean("A footnote.1")
        assert cleaned == "A footnote."
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
        assert classification == "body"

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

    # --- Not suspicious: OCR artifacts correctly identified ---

    def test_identical_text_not_suspicious(self) -> None:
        assert _has_suspicious_substitutions("Hello world.", "Hello world.") is False

    def test_embedded_pipe_char_is_artifact_not_suspicious(self) -> None:
        """Regression: old re.sub stripped 'a|nd' → 'and' (valid), wrongly flagging it.
        New strip-only approach leaves '|' embedded, keeping it invalid."""
        assert _has_suspicious_substitutions(
            "He said a|nd walked away.",
            "He said and walked away."
        ) is False

    def test_embedded_angle_bracket_artifact_not_suspicious(self) -> None:
        """'t<;' has embedded '<' — strip only removes boundary ';', leaving 't<' (invalid)."""
        assert _has_suspicious_substitutions(
            "He wishes t<; thank her.",
            "He wishes to thank her."
        ) is False

    def test_pure_ocr_garbage_replaced_with_valid_word_not_suspicious(self) -> None:
        """Unrecognizable OCR token replaced with a valid word is a legitimate fix."""
        assert _has_suspicious_substitutions(
            "The xzqpf was undeniable.",
            "The truth was undeniable."
        ) is False

    # --- Not suspicious: boundary punctuation handled correctly ---

    def test_boundary_comma_stripped_same_word_not_suspicious(self) -> None:
        """'council,' and 'council' both strip to 'council' — treated as equal."""
        assert _has_suspicious_substitutions(
            "The council, agreed on the plan.",
            "The council agreed on the plan."
        ) is False

    def test_boundary_parentheses_stripped_same_word_not_suspicious(self) -> None:
        """'(word)' strips to 'word' — same as cleaned 'word'."""
        assert _has_suspicious_substitutions(
            "The (government) responded.",
            "The government responded."
        ) is False

    def test_trailing_period_stripped_same_word_not_suspicious(self) -> None:
        assert _has_suspicious_substitutions(
            "She agreed.",
            "She agreed."
        ) is False

    # --- Not suspicious: valid-word substitutions are handled by _restore_valid_words ---

    def test_valid_word_swapped_for_different_valid_word_not_suspicious(self) -> None:
        """'cat' → 'dog' is now handled by _restore_valid_words, not flagged here."""
        assert _has_suspicious_substitutions(
            "The cat sat on the mat.",
            "The dog sat on the mat."
        ) is False

    def test_valid_word_replaced_with_invalid_word_not_suspicious(self) -> None:
        """valid → invalid is caught by _restore_valid_words (restores original), not flagged here."""
        assert _has_suspicious_substitutions(
            "The quick brown fox.",
            "The quick xzqpf fox."
        ) is False

    # --- Not suspicious: hyphen/em-dash equivalence ---

    def test_hyphen_to_em_dash_in_compound_not_suspicious(self) -> None:
        """'work-far' → 'work—far': same token after dash normalization — not suspicious."""
        assert _has_suspicious_substitutions(
            "a closely-integrated work-far exceeding expectations",
            "a closely-integrated work—far exceeding expectations"
        ) is False

    # --- Not suspicious: missing hyphen restored ---

    def test_missing_hyphen_restored_not_suspicious(self) -> None:
        """'wellknown' → 'well-known': equal after hyphen stripping — not suspicious."""
        assert _has_suspicious_substitutions(
            "their wellknown paradoxes",
            "their well-known paradoxes"
        ) is False

    # --- Not suspicious: diacritics restored ---

    def test_diacritics_restored_not_suspicious(self) -> None:
        """'Eotvos' → 'Eötvös': equal after ASCII-folding — not suspicious."""
        assert _has_suspicious_substitutions(
            "the experiments by Eotvos more recently",
            "the experiments by Eötvös more recently"
        ) is False

    # --- Not suspicious: symbol/punctuation substitution ---

    def test_symbol_substitution_not_suspicious(self) -> None:
        """'star:✦' → 'star:*': non-alphabetic tokens are skipped — not suspicious."""
        assert _has_suspicious_substitutions(
            "marked with a star:✶",
            "marked with a star:*"
        ) is False

    def test_pure_symbol_token_not_suspicious(self) -> None:
        """A standalone non-alphanumeric token replaced by any other token is not suspicious."""
        assert _has_suspicious_substitutions(
            "marked with a star ✶ here",
            "marked with a star * here"
        ) is False

    def test_multi_char_symbol_token_not_suspicious(self) -> None:
        """A multi-character symbolic token with no letters or digits is not suspicious."""
        assert _has_suspicious_substitutions(
            "the symbol ✶✶ appears",
            "the symbol ** appears"
        ) is False

    # --- Not suspicious: invalid→invalid is now trusted ---

    def test_invalid_to_invalid_is_trusted_not_suspicious(self) -> None:
        """Both original and replacement are invalid — we trust the LLM to do its best.
        Previously this was flagged as suspicious, but rejecting invalid→invalid caused
        whole paragraphs to fail when the LLM made an imperfect but reasonable substitution
        (e.g. 'heiden' → 'beiden' in a German title, or any OCR artifact the LLM
        partially corrects). We now accept the LLM's version in this case."""
        assert _has_suspicious_substitutions(
            "The xzqpf was clear.",
            "The zqpfx was clear."
        ) is False

    def test_ocr_artifact_replaced_with_valid_word_not_suspicious(self) -> None:
        """'xzqpf' → 'truth': OCR fixed correctly — not suspicious."""
        assert _has_suspicious_substitutions(
            "The xzqpf was undeniable.",
            "The truth was undeniable."
        ) is False


# --- TestRestoreListPrefix ---

class TestRestoreListPrefix:
    def test_parenthesized_number_restored_when_dropped(self) -> None:
        """(3) prefix dropped by LLM is restored."""
        result = _restore_list_prefix(
            "(3) All human actions are egotistic.",
            "All human actions are egotistic."
        )
        assert result == "(3) All human actions are egotistic."

    def test_dot_number_prefix_restored_when_dropped(self) -> None:
        """'3. ' prefix dropped by LLM is restored."""
        result = _restore_list_prefix(
            "3. All human actions are egotistic.",
            "All human actions are egotistic."
        )
        assert result == "3. All human actions are egotistic."

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
    def test_em_dash_replaced_with_hyphen(self) -> None:
        assert _normalize_dashes('well—known') == 'well-known'

    def test_en_dash_replaced_with_hyphen(self) -> None:
        assert _normalize_dashes('well–known') == 'well-known'

    def test_plain_hyphen_unchanged(self) -> None:
        assert _normalize_dashes('well-known') == 'well-known'

    def test_no_dash_unchanged(self) -> None:
        assert _normalize_dashes('hello') == 'hello'

    def test_both_em_and_en_dash_replaced(self) -> None:
        assert _normalize_dashes('a—b–c') == 'a-b-c'

    def test_empty_string(self) -> None:
        assert _normalize_dashes('') == ''


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
        """em-dash variant of a hyphenated compound should not be restored."""
        result = _restore_valid_words(
            "a false-as assumption",
            "a false—as assumption"
        )
        # "false-as" and "false—as" normalize to the same thing — no substitution
        assert result == "a false—as assumption"

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


# --- TestCoerceClassification ---

class TestCoerceClassification:

    def test_footnote_hint_returns_footnote(self) -> None:
        assert _coerce_classification('footnote') == 'footnote'

    def test_endnote_hint_returns_footnote(self) -> None:
        assert _coerce_classification('endnote') == 'footnote'

    def test_note_hint_returns_footnote(self) -> None:
        assert _coerce_classification('note') == 'footnote'

    def test_body_hint_returns_body(self) -> None:
        assert _coerce_classification('body') == 'body'

    def test_main_hint_returns_body(self) -> None:
        assert _coerce_classification('main content') == 'body'

    def test_prose_hint_returns_body(self) -> None:
        assert _coerce_classification('prose') == 'body'

    def test_index_hint_returns_drop(self) -> None:
        assert _coerce_classification('index') == 'drop'

    def test_bibliography_hint_returns_drop(self) -> None:
        assert _coerce_classification('bibliograph') == 'drop'

    def test_reference_hint_returns_drop(self) -> None:
        assert _coerce_classification('reference list') == 'drop'

    def test_matching_is_case_insensitive(self) -> None:
        assert _coerce_classification('FOOTNOTE') == 'footnote'
        assert _coerce_classification('BODY TEXT') == 'body'
        assert _coerce_classification('INDEX') == 'drop'

    def test_unknown_label_returns_none(self) -> None:
        assert _coerce_classification('something_unknown') is None

    def test_empty_string_returns_none(self) -> None:
        assert _coerce_classification('') is None


# --- TestIntegration ---

class TestIntegration:
    @pytest.mark.integration
    def test_real_llm_call_body(self) -> None:
        """Integration test — requires a running LLM with llama3.1:8b."""
        cleaner = make_cleaner()
        paragraph = "This is a sample paragraph from a book about philosophy and rationality."
        cleaned, classification = cleaner.clean(paragraph)
        assert classification == 'body'
        assert "philosophy" in cleaned
        assert "rationality" in cleaned
        assert cleaned == paragraph

    @pytest.mark.integration
    def test_real_llm_call_footnote(self) -> None:
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
    def test_real_llm_call_drop(self) -> None:
        """Integration test — requires a running LLM with llama3.1:8b."""
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
