"""Tests for the TextProcessor class and _all_words_valid helper."""

import pytest
from unittest.mock import MagicMock, ANY
from text_chunk import RawChunk
from text_cleaner import FormulaMode
from text_processor import TextProcessor, _all_words_valid


# --- Fixtures ---

def make_chunk(text: str, label: str = 'text', page: str = '') -> RawChunk:
    """Create a RawChunk with the given text, label, and optional page number."""
    meta = {'page_#': page} if page else {}
    return RawChunk(text=text, meta=meta, label=label)


def make_processor(min_paragraph_size: int = 0,
                   include_footnotes: bool = False,
                   strip_footnote_markers: bool = True) -> TextProcessor:
    """Create a TextProcessor with the given settings."""
    return TextProcessor(min_paragraph_size=min_paragraph_size,
                         include_footnotes=include_footnotes,
                         strip_footnote_markers=strip_footnote_markers)


def make_formula_cleaner(ocr_result: str = 'cleaned formula',
                          audio_result: str = 'spoken formula',
                          formula_mode: FormulaMode = FormulaMode.CLEAN,
                          body_result: str | None = None) -> MagicMock:
    """Create a mock TextCleaner with formula cleaning methods and a formula_mode attribute.

    body_result: if set, wires up clean() to return (body_result, 'body') for use
    in FormulaMode.NONE tests where formula chunks fall through to the body-text path.
    """
    cleaner = MagicMock()
    cleaner.formula_mode = formula_mode
    cleaner.clean_formula_ocr.side_effect = lambda formula, page_context='': ocr_result
    cleaner.clean_formula.side_effect = lambda formula, page_context='': audio_result
    # clean(formula=True) is now the OCR pass for CLEAN/AUDIO modes
    cleaner.clean.side_effect = lambda text, page_context='', formula=False: (ocr_result, 'body')
    if body_result is not None:
        # NONE mode: formula chunks fall through to body-text path without formula=True
        cleaner.clean.side_effect = lambda text, page_context='': (body_result, 'body')
    return cleaner


def make_cleaner(classification: str = 'body', cleaned: str | None = None) -> MagicMock:
    """Create a mock TextCleaner that returns the given classification."""
    cleaner = MagicMock()
    cleaner.clean.side_effect = lambda text, page_context='': (
        cleaned if cleaned is not None else text, classification
    )
    return cleaner


# --- TestProcess ---

class TestProcess:
    def test_empty_chunks_returns_empty(self) -> None:
        processor = make_processor()
        result = processor.process([])
        assert result == []

    def test_single_complete_sentence(self) -> None:
        processor = make_processor()
        chunks = [make_chunk("This is a complete sentence.")]
        result = processor.process(chunks)
        assert len(result) == 1
        assert result[0].text == "This is a complete sentence."

    def test_incomplete_sentence_accumulated_with_next(self) -> None:
        processor = make_processor()
        chunks = [
            make_chunk("This is incomplete"),
            make_chunk("and this completes it."),
        ]
        result = processor.process(chunks)
        assert len(result) == 1
        assert "This is incomplete" in result[0].text
        assert "and this completes it." in result[0].text

    def test_incomplete_sentence_at_end_still_emitted(self) -> None:
        processor = make_processor()
        chunks = [make_chunk("This is incomplete")]
        result = processor.process(chunks)
        assert len(result) == 1
        assert "This is incomplete" in result[0].text

    def test_short_paragraphs_accumulated_until_min_size(self) -> None:
        processor = make_processor(min_paragraph_size=100)
        chunks = [
            make_chunk("First sentence."),
            make_chunk("Second sentence."),
            make_chunk("Third sentence which finally makes it long enough to emit."),
        ]
        result = processor.process(chunks)
        assert len(result) == 1
        assert "First sentence." in result[0].text
        assert "Second sentence." in result[0].text

    def test_section_header_emitted_as_own_paragraph(self) -> None:
        processor = make_processor()
        chunks = [make_chunk("Chapter One", label='section_header')]
        result = processor.process(chunks)
        assert len(result) == 1
        assert result[0].text == "Chapter One"

    def test_section_header_flushes_accumulated_paragraph(self) -> None:
        processor = make_processor(min_paragraph_size=100)
        chunks = [
            make_chunk("Accumulated text."),
            make_chunk("Chapter One", label='section_header'),
        ]
        result = processor.process(chunks)
        assert any("Accumulated text." in r.text for r in result)
        assert any("Chapter One" in r.text for r in result)

    def test_section_header_resets_accumulator(self) -> None:
        processor = make_processor(min_paragraph_size=100)
        chunks = [
            make_chunk("Before header."),
            make_chunk("Chapter One", label='section_header'),
            make_chunk("After header."),
        ]
        result = processor.process(chunks)
        assert any("Before header." in r.text for r in result)
        assert any("Chapter One" in r.text for r in result)
        assert any("After header." in r.text for r in result)

    def test_section_name_in_meta_after_header(self) -> None:
        processor = make_processor()
        chunks = [
            make_chunk("Chapter One", label='section_header'),
            make_chunk("Content here."),
        ]
        result = processor.process(chunks)
        content = next(r for r in result if "Content here." in r.text)
        assert content.meta["section_name"] == "Chapter One"

    def test_page_header_skipped(self) -> None:
        processor = make_processor()
        chunks = [
            make_chunk("Page Header", label='page_header'),
            make_chunk("Real content."),
        ]
        result = processor.process(chunks)
        assert all("Page Header" not in r.text for r in result)

    def test_page_footer_skipped(self) -> None:
        processor = make_processor()
        chunks = [
            make_chunk("Real content."),
            make_chunk("Page Footer", label='page_footer'),
        ]
        result = processor.process(chunks)
        assert all("Page Footer" not in r.text for r in result)

    def test_footnote_excluded_by_default(self) -> None:
        processor = make_processor(include_footnotes=False)
        chunks = [
            make_chunk("Main text."),
            make_chunk("Footnote text.", label='footnote'),
        ]
        result = processor.process(chunks)
        assert all("Footnote text." not in r.text for r in result)

    def test_footnote_included_when_flag_set(self) -> None:
        processor = make_processor(include_footnotes=True)
        chunks = [
            make_chunk("Main text."),
            make_chunk("Footnote text.", label='footnote'),
        ]
        result = processor.process(chunks)
        assert any("Footnote text." in r.text for r in result)

    def test_paragraph_numbers_increment(self) -> None:
        processor = make_processor()
        chunks = [
            make_chunk("First paragraph."),
            make_chunk("Second paragraph."),
        ]
        result = processor.process(chunks)
        assert result[0].meta["paragraph_#"] == "1"
        assert result[1].meta["paragraph_#"] == "2"

    def test_meta_passed_through(self) -> None:
        processor = make_processor()
        chunk = RawChunk(text="Some content.", meta={"source": "test"}, label='text')
        result = processor.process([chunk])
        assert result[0].meta["source"] == "test"

    def test_next_section_header_forces_emit(self) -> None:
        processor = make_processor(min_paragraph_size=1000)
        chunks = [
            make_chunk("Short paragraph."),
            make_chunk("Chapter Two", label='section_header'),
        ]
        result = processor.process(chunks)
        assert any("Short paragraph." in r.text for r in result)

    def test_generate_text_file_creates_file(self, tmp_path) -> None:
        processor = make_processor()
        chunks = [make_chunk("Some content.")]
        output_path = tmp_path / "test"
        processor.process(chunks, output_path=output_path, generate_text_file=True)
        assert (tmp_path / "test_processed_paragraphs.txt").exists()

    def test_generate_text_file_content(self, tmp_path) -> None:
        processor = make_processor()
        chunks = [make_chunk("Some content.")]
        output_path = tmp_path / "test"
        processor.process(chunks, output_path=output_path, generate_text_file=True)
        content = (tmp_path / "test_processed_paragraphs.txt").read_text(encoding="utf-8")
        assert "Some content." in content

    def test_generate_text_file_false_does_not_create_file(self, tmp_path) -> None:
        processor = make_processor()
        chunks = [make_chunk("Some content.")]
        output_path = tmp_path / "test"
        processor.process(chunks, output_path=output_path, generate_text_file=False)
        assert not (tmp_path / "test_processed_paragraphs.txt").exists()

    def test_page_break_hyphen_joined_before_cleaning(self) -> None:
        """A hyphenated word split by a page break is joined before any other processing."""
        processor = make_processor()
        # "character" with a soft hyphen in it in a single chunk simulates a page-break OCR artifact
        # noinspection SpellCheckingInspection
        chunks = [make_chunk("an historical charac\u00adter.")]
        result = processor.process(chunks)
        assert result[0].text == "an historical character."

    def test_process_can_be_called_multiple_times(self) -> None:
        processor = make_processor()
        chunks = [make_chunk("First run.")]
        result1 = processor.process(chunks)
        result2 = processor.process(chunks)
        assert len(result1) == 1
        assert len(result2) == 1
        assert result1[0].text == result2[0].text


# --- TestCleaner ---

# Note: several tests below use inputs containing digits or misspelled words.
# This is intentional — _all_words_valid() skips the cleaner for clean text,
# so inputs must contain an artifact to ensure TextProcessor actually invokes
# TextCleaner. The artifact is a precondition, not the subject of the test.

class TestCleaner:
    def test_cleaner_body_paragraph_kept(self) -> None:
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk("Some b0dy text.")])  # digit forces cleaner call
        assert len(result) == 1
        assert "Some b0dy text." in result[0].text

    def test_cleaner_drop_paragraph_discarded(self) -> None:
        cleaner = make_cleaner(classification='drop')
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk("Table of contents ... 1")])  # digit forces cleaner call
        assert result == []

    def test_cleaner_footnote_excluded_by_default(self) -> None:
        cleaner = make_cleaner(classification='footnote')
        processor = TextProcessor(cleaner=cleaner, include_footnotes=False)
        result = processor.process([make_chunk("1 A footnote.")])  # digit forces cleaner call
        assert result == []

    def test_cleaner_footnote_included_when_flag_set(self) -> None:
        cleaner = make_cleaner(classification='footnote', cleaned="A footnote.")
        processor = TextProcessor(cleaner=cleaner, include_footnotes=True)
        result = processor.process([make_chunk("1 A footnote.")])  # digit forces cleaner call
        assert len(result) == 1
        assert result[0].is_footnote

    def test_cleaner_cleaned_text_used(self) -> None:
        cleaner = make_cleaner(classification='body', cleaned="Fixed text.")
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk("Brok en text.")])  # misspelling forces cleaner call
        assert result[0].text == "Fixed text."

    def test_cleaner_receives_page_context(self) -> None:
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        chunks = [
            make_chunk("F1rst paragraph.", page='1'),   # digit forces cleaner call
            make_chunk("S3cond paragraph.", page='1'),  # digit forces cleaner call
        ]
        processor.process(chunks)
        call_args = cleaner.clean.call_args
        assert call_args[1]['page_context'] != '' or call_args[0][1] != ''

    def test_cleaner_called_once_per_flushed_paragraph(self) -> None:
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        chunks = [make_chunk("F1rst."), make_chunk("S3cond.")]  # digits force cleaner calls
        processor.process(chunks)
        assert cleaner.clean.call_count == 2

    def test_no_cleaner_uses_existing_behavior(self) -> None:
        processor = make_processor()
        chunks = [make_chunk("Normal text.")]
        result = processor.process(chunks)
        assert len(result) == 1
        assert result[0].text == "Normal text."


# --- TestFormulaMode ---

class TestFormulaMode:
    """Tests for formula_mode parameter controlling how formula chunks are handled."""

    def test_no_cleaner_emits_raw_text(self) -> None:
        """Without a cleaner, formula text is emitted unchanged."""
        processor = TextProcessor()
        result = processor.process([make_chunk('x + y = z', label='formula')])
        assert len(result) == 1
        assert result[0].text == 'x + y = z'

    def test_clean_calls_clean_and_emits_result(self) -> None:
        """CLEAN mode calls clean(formula=True) and emits its result."""
        cleaner = make_formula_cleaner(ocr_result='x + y = z', audio_result='x plus y',
                                       formula_mode=FormulaMode.CLEAN)
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk('x -+- y == z (garbled)', label='formula')])
        assert len(result) == 1
        assert result[0].text == 'x + y = z'
        cleaner.clean.assert_called_once()
        cleaner.clean_formula_ocr.assert_not_called()
        cleaner.clean_formula.assert_not_called()

    def test_audio_calls_both_passes_in_order(self) -> None:
        """AUDIO mode calls clean(formula=True) then clean_formula."""
        cleaner = make_formula_cleaner(ocr_result='x + y = z', audio_result='x plus y equals z',
                                       formula_mode=FormulaMode.AUDIO)
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk('x -+- y == z (garbled)', label='formula')])
        assert len(result) == 1
        assert result[0].text == 'x plus y equals z'
        cleaner.clean.assert_called_once()
        cleaner.clean_formula.assert_called_once()
        cleaner.clean_formula_ocr.assert_not_called()

    def test_audio_feeds_clean_output_into_audio_pass(self) -> None:
        """AUDIO mode must pass clean(formula=True)'s result into clean_formula, not the raw text."""
        cleaner = make_formula_cleaner(ocr_result='x² + y² = z²',
                                       audio_result='x squared plus y squared equals z squared',
                                       formula_mode=FormulaMode.AUDIO)
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk('x2 + y2 = z2 (OCR mess)', label='formula')])
        cleaner.clean_formula.assert_called_once_with('x² + y² = z²', page_context=ANY)
        cleaner.clean_formula_ocr.assert_not_called()


# --- TestFormulaAnnotation ---

class TestFormulaAnnotation:
    """Tests for verbose annotation of formula chunks."""

    def test_clean_mode_prints_annotation(self, capsys) -> None:
        """In CLEAN mode with verbose=True, annotation shows original and cleaned text."""
        cleaner = make_formula_cleaner(ocr_result='x + y = z', formula_mode=FormulaMode.CLEAN)
        processor = TextProcessor(cleaner=cleaner, verbose=True)
        processor.process([make_chunk('x -+- y == z', label='formula')])
        out = capsys.readouterr().out
        assert '[FORMULA]' in out
        assert 'x -+- y == z' in out
        assert 'x + y = z' in out

    def test_audio_mode_annotation_includes_audio_text(self, capsys) -> None:
        """In AUDIO mode with verbose=True, annotation also shows the final audio text."""
        cleaner = make_formula_cleaner(ocr_result='x + y = z', audio_result='x plus y equals z',
                                       formula_mode=FormulaMode.AUDIO)
        processor = TextProcessor(cleaner=cleaner, verbose=True)
        processor.process([make_chunk('x -+- y == z', label='formula')])
        out = capsys.readouterr().out
        assert 'x plus y equals z' in out

    def test_not_verbose_no_annotation(self, capsys) -> None:
        """Without verbose=True, no formula annotation is printed."""
        cleaner = make_formula_cleaner(ocr_result='x + y = z', formula_mode=FormulaMode.CLEAN)
        processor = TextProcessor(cleaner=cleaner, verbose=False)
        processor.process([make_chunk('x -+- y == z', label='formula')])
        out = capsys.readouterr().out
        assert '[FORMULA]' not in out


# --- TestFormulaModeNoneRouting ---

class TestFormulaModeNoneRouting:
    """Tests that FormulaMode.NONE routes formula chunks through the body-text path.

    CLEAN/AUDIO modes send formula chunks to _handle_formula (calling clean_formula_ocr).
    NONE mode lets them fall through to _process_chunk (calling clean() like body text).
    """

    def test_none_mode_does_not_call_clean_formula_ocr(self) -> None:
        """NONE mode never calls clean_formula_ocr on a formula chunk."""
        cleaner = make_formula_cleaner(formula_mode=FormulaMode.NONE,
                                       body_result='x + y = z')
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk('x -+- y == z (garbled)', label='formula')])
        cleaner.clean_formula_ocr.assert_not_called()

    def test_none_mode_does_not_call_clean_formula(self) -> None:
        """NONE mode never calls clean_formula on a formula chunk."""
        cleaner = make_formula_cleaner(formula_mode=FormulaMode.NONE,
                                       body_result='x + y = z')
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk('x -+- y == z (garbled)', label='formula')])
        cleaner.clean_formula.assert_not_called()

    def test_none_mode_calls_clean_on_formula_chunk(self) -> None:
        """NONE mode sends formula chunks through clean(), the same as body text."""
        cleaner = make_formula_cleaner(formula_mode=FormulaMode.NONE,
                                       body_result='x + y = z')
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk('x -+- y == z (garbled)', label='formula')])
        cleaner.clean.assert_called()

    def test_none_mode_result_not_labeled_formula(self) -> None:
        """NONE mode emits formula chunks with a body-text label, not 'formula'."""
        cleaner = make_formula_cleaner(formula_mode=FormulaMode.NONE,
                                       body_result='The probability is high.')
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk('The probability is high.', label='formula')])
        assert len(result) == 1
        assert result[0].label != 'formula'

    def test_none_mode_no_formula_annotation(self, capsys) -> None:
        """NONE mode never prints [FORMULA] even with verbose=True."""
        cleaner = make_formula_cleaner(formula_mode=FormulaMode.NONE,
                                       body_result='The probability is high.')
        processor = TextProcessor(cleaner=cleaner, verbose=True)
        processor.process([make_chunk('The probability is high.', label='formula')])
        out = capsys.readouterr().out
        assert '[FORMULA]' not in out

    def test_clean_mode_calls_clean_with_formula_true(self) -> None:
        """Contrast: CLEAN mode calls clean(formula=True), not the body-text path."""
        cleaner = make_formula_cleaner(ocr_result='x + y = z', formula_mode=FormulaMode.CLEAN)
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk('x -+- y == z', label='formula')])
        cleaner.clean.assert_called_once()
        cleaner.clean_formula_ocr.assert_not_called()


# --- TestAllWordsValid ---

class TestAllWordsValid:
    @pytest.mark.parametrize("text", [
        "The dog ran quickly",  # common words
        "",                     # empty string — no invalid tokens
        "Hello, world.",        # punctuation stripped before check
        "I saw a dog",          # 'a' and 'I' are valid single-letter words
    ])
    def test_returns_true(self, text: str) -> None:
        assert _all_words_valid(text) is True

    @pytest.mark.parametrize("text", [
        "I am hppy today",        # non-word OCR artifact
        "Chapter 1789",           # standalone number
        "The dog ran quickly1.",  # digit embedded in word
    ])
    def test_returns_false(self, text: str) -> None:
        assert _all_words_valid(text) is False

    def test_single_letter_fragment_returns_false(self) -> None:
        """A lone single-letter token (not 'a' or 'i') is a line-break artifact.

        OCR sometimes splits a word across a line so the last letter of the
        previous line becomes a standalone token: 'p referring' for 'preferring'.
        Because every letter is a valid word in NLTK, _all_words_valid would
        otherwise pass the paragraph and skip the LLM, leaving the fragment
        unjoined.  Only 'a' and 'i' are legitimate standalone single-letter
        English words.
        """
        assert _all_words_valid("p referring") is False
        assert _all_words_valid("the s cientific method") is False


# --- TestSkipCleanerWhenAllWordsValid ---

class TestSkipCleanerWhenAllWordsValid:
    def test_cleaner_not_called_when_all_words_valid(self) -> None:
        """Cleaner should not be called when every word in the paragraph is valid."""
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk("The dog ran quickly.")])
        cleaner.clean.assert_not_called()

    def test_paragraph_returned_unchanged_when_all_words_valid(self) -> None:
        """Paragraph should pass through unmodified when all words are valid."""
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk("The dog ran quickly.")])
        assert result[0].text == "The dog ran quickly."

    def test_cleaner_called_when_invalid_word_present(self) -> None:
        """Cleaner should be called when the paragraph contains an invalid word."""
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        # noinspection SpellCheckingInspection
        processor.process([make_chunk("The dog ran qukckly.")])
        cleaner.clean.assert_called_once()

    def test_cleaner_called_when_digit_present(self) -> None:
        """Cleaner should be called when the paragraph contains a digit token."""
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk("See footnote 4 for details.")])
        cleaner.clean.assert_called_once()


# --- TestShouldAccumulate ---

class TestShouldAccumulate:

    def test_no_next_chunk_returns_false(self) -> None:
        """End of document — always emit regardless of content."""
        processor = make_processor()
        assert processor._should_accumulate("Incomplete sentence", None) is False

    def test_no_sentence_end_with_next_returns_true(self) -> None:
        """Incomplete paragraph must accumulate when more text is coming."""
        processor = make_processor()
        next_chunk = make_chunk("More text.")
        assert processor._should_accumulate("Incomplete sentence", next_chunk) is True

    def test_sentence_end_next_is_section_header_returns_false(self) -> None:
        """Complete paragraph should emit before a section boundary."""
        processor = make_processor(min_paragraph_size=1000)
        next_chunk = make_chunk("Chapter Two", label='section_header')
        assert processor._should_accumulate("Complete sentence.", next_chunk) is False

    def test_sentence_end_next_is_not_body_text_returns_false(self) -> None:
        """Complete paragraph should emit when next chunk is not body text."""
        processor = make_processor(min_paragraph_size=1000)
        next_chunk = make_chunk("104", label='page_header')
        assert processor._should_accumulate("Complete sentence.", next_chunk) is False

    def test_sentence_end_size_reached_returns_false(self) -> None:
        """Complete paragraph at or above min size should emit."""
        processor = make_processor(min_paragraph_size=10)
        next_chunk = make_chunk("More text.")
        assert processor._should_accumulate("Complete sentence.", next_chunk) is False

    def test_sentence_end_below_min_size_returns_true(self) -> None:
        """Complete but short paragraph should accumulate when more body text follows."""
        processor = make_processor(min_paragraph_size=10000)
        next_chunk = make_chunk("More text.")
        assert processor._should_accumulate("Short.", next_chunk) is True

    def test_accumulated_paragraph_counted_toward_min_size(self) -> None:
        """Already-accumulated text contributes to the size check."""
        processor = make_processor(min_paragraph_size=20)
        processor._paragraph = ["Already accumulated text here."]
        next_chunk = make_chunk("More text.")
        # combined_count (30) + len("Done.") (5) >= 20 → emit
        assert processor._should_accumulate("Done.", next_chunk) is False


# --- TestBuildMeta ---

class TestBuildMeta:

    def test_includes_paragraph_number(self) -> None:
        processor = make_processor()
        processor._para_num = 7
        result = processor._build_meta({})
        assert result['paragraph_#'] == '7'

    def test_passes_through_base_meta(self) -> None:
        processor = make_processor()
        processor._para_num = 1
        result = processor._build_meta({'page_#': '42', 'source': 'test.pdf'})
        assert result['page_#'] == '42'
        assert result['source'] == 'test.pdf'

    def test_includes_section_name_when_set(self) -> None:
        processor = make_processor()
        processor._para_num = 1
        processor._section_name = "Chapter One"
        result = processor._build_meta({})
        assert result['section_name'] == 'Chapter One'

    def test_omits_section_name_when_empty(self) -> None:
        processor = make_processor()
        processor._para_num = 1
        processor._section_name = ""
        result = processor._build_meta({})
        assert 'section_name' not in result

    def test_does_not_mutate_base_meta(self) -> None:
        processor = make_processor()
        processor._para_num = 1
        base = {'page_#': '1'}
        processor._build_meta(base)
        assert 'paragraph_#' not in base


# --- TestBuildPageContexts ---

class TestBuildPageContexts:

    def test_groups_chunks_by_page(self) -> None:
        chunks = [
            make_chunk("First sentence.", page='1'),
            make_chunk("Second sentence.", page='2'),
        ]
        result = TextProcessor._build_page_contexts(chunks)
        assert '1' in result
        assert '2' in result
        assert 'First sentence.' in result['1']
        assert 'Second sentence.' in result['2']

    def test_multiple_chunks_on_same_page_joined(self) -> None:
        chunks = [
            make_chunk("Sentence one.", page='3'),
            make_chunk("Sentence two.", page='3'),
        ]
        result = TextProcessor._build_page_contexts(chunks)
        assert 'Sentence one.' in result['3']
        assert 'Sentence two.' in result['3']

    def test_empty_chunk_list_returns_empty_dict(self) -> None:
        assert TextProcessor._build_page_contexts([]) == {}

    def test_chunks_without_page_number_are_ignored(self) -> None:
        chunks = [make_chunk("No page.")]
        result = TextProcessor._build_page_contexts(chunks)
        assert result == {}


def make_chunk_with_pages(text: str, page_label: str, physical_page: str) -> RawChunk:
    """Create a RawChunk with both page_# (label) and physical_page_# metadata."""
    return RawChunk(
        text=text,
        meta={'page_#': page_label, 'physical_page_#': physical_page},
        label='text',
    )


# --- TestReportPageProgress ---

class TestReportPageProgress:
    """Tests for TextProcessor._report_page_progress.

    In verbose mode, progress is buffered in _pending_header rather than
    printed immediately, and only flushed when something noteworthy happens
    on that page (e.g. the LLM is invoked).  In non-verbose mode, progress
    is still printed immediately at every 10-page boundary.

    The milestone calculation must use physical_page_# (always an integer)
    so that Roman numeral PDF labels don't silently suppress reporting.  The
    display should show the PDF label when it differs from the physical number.
    """

    def test_roman_numeral_label_still_reports_using_physical(self) -> None:
        """When page_# is a Roman numeral, physical_page_# drives reporting."""
        processor = TextProcessor(verbose=True)
        processor._init_state()
        chunk = make_chunk_with_pages("Text.", page_label='i', physical_page='1')
        processor._report_page_progress(chunk)
        assert '[Page' in processor._pending_header

    def test_matching_label_pending_header_contains_single_page_number(self) -> None:
        """When label equals physical, pending header contains '[Page N]'."""
        processor = TextProcessor(verbose=True)
        processor._init_state()
        chunk = make_chunk_with_pages("Text.", page_label='42', physical_page='42')
        processor._report_page_progress(chunk)
        assert '[Page 42]' in processor._pending_header
        assert '/' not in processor._pending_header

    def test_differing_label_pending_header_contains_both(self) -> None:
        """When label differs from physical, pending header contains both."""
        processor = TextProcessor(verbose=True)
        processor._init_state()
        chunk = make_chunk_with_pages("Text.", page_label='1', physical_page='41')
        processor._report_page_progress(chunk)
        assert '[Page 1 / Page 41]' in processor._pending_header

    def test_no_physical_falls_back_to_page_label(self) -> None:
        """Without physical_page_#, falls back to page_# for backward compatibility."""
        processor = TextProcessor(verbose=True)
        processor._init_state()
        chunk = make_chunk("Complete sentence.", page='7')
        processor._report_page_progress(chunk)
        assert '[Page 7]' in processor._pending_header

    def test_non_verbose_never_sets_pending_header(self) -> None:
        """In non-verbose mode, _report_page_progress never sets a pending header."""
        processor = TextProcessor(verbose=False)
        processor._init_state()
        chunk = make_chunk_with_pages("Text.", page_label='10', physical_page='10')
        processor._report_page_progress(chunk)
        assert processor._pending_header == ""

    def test_same_page_not_pending_twice_verbose(self) -> None:
        """The same physical page only sets the pending header once."""
        processor = TextProcessor(verbose=True)
        processor._init_state()
        chunk = make_chunk_with_pages("Text.", page_label='5', physical_page='5')
        processor._report_page_progress(chunk)
        first_pending = processor._pending_header
        processor._report_page_progress(chunk)
        assert processor._pending_header == first_pending

    def test_page_header_not_printed_when_all_words_valid(self, capsys) -> None:
        """Page header is not printed when the cleaner is skipped (all words valid)."""
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner, verbose=True)
        chunk = make_chunk_with_pages("The dog ran quickly.", page_label='5', physical_page='5')
        processor.process([chunk])
        out = capsys.readouterr().out
        assert '[Page' not in out

    def test_page_header_flushed_before_noteworthy_output(self, capsys) -> None:
        """Page header is printed before LLM output when the cleaner is invoked."""
        cleaner = make_cleaner(classification='drop')
        processor = TextProcessor(cleaner=cleaner, verbose=True)
        chunk = make_chunk_with_pages("Table of c0ntents.", page_label='5', physical_page='5')
        processor.process([chunk])
        out = capsys.readouterr().out
        assert '[Page' in out
        assert out.index('[Page') < out.index('[LLM DROP]')


# --- TestUprfontFootnoteReclassification ---

class TestUpfrontFootnoteReclassification:
    """Chunks whose text starts with 1-2 digits immediately followed by an uppercase
    letter are reclassified as footnotes during the upfront preprocessing pass."""

    @pytest.mark.parametrize("text", [
        "3See my Poverty of Historicism, section 32.",  # digit+uppercase (1 digit)
        "14Cf. the earlier discussion on page 42.",     # digit+uppercase (2 digits)
    ])
    def test_digit_uppercase_chunk_is_dropped(self, text: str) -> None:
        """A 'text' chunk matching the digit+uppercase pattern is reclassified and dropped."""
        processor = make_processor(include_footnotes=False)
        result = processor.process([make_chunk(text)])
        assert result == []

    @pytest.mark.parametrize("text", [
        "3 Some body text that is complete.",      # space between digit and letter
        "1st place goes to the fastest runner.",   # lowercase after digit
    ])
    def test_digit_chunk_is_kept(self, text: str) -> None:
        """Chunks that don't match the footnote pattern are kept as body text."""
        processor = make_processor(include_footnotes=False)
        result = processor.process([make_chunk(text)])
        assert len(result) == 1

    def test_reclassified_footnote_included_when_flag_set(self) -> None:
        """If include_footnotes=True, a reclassified chunk still appears in output."""
        processor = make_processor(include_footnotes=True)
        chunks = [make_chunk("3See my Poverty of Historicism, section 32.")]
        result = processor.process(chunks)
        assert len(result) == 1


class TestFootnoteChunkNotStripped:
    """Footnote chunks must not have attached numbers stripped — they contain
    bibliographic references like 'p.14' or nested footnote markers like '.4'
    that look identical to the pattern we strip from body text."""

    def test_attached_number_in_footnote_chunk_not_stripped(self) -> None:
        chunks = [
            make_chunk("Main body paragraph text."),
            make_chunk("Jones (1987).4 For a complete discussion.", label='footnote'),
        ]
        processor = make_processor(include_footnotes=True)
        result = processor.process(chunks)
        footnote_results = [r for r in result if r.label == 'footnote']
        assert len(footnote_results) == 1
        assert "1987).4" in footnote_results[0].text

    def test_page_citation_in_footnote_chunk_not_stripped(self) -> None:
        chunks = [
            make_chunk("Main body paragraph text."),
            make_chunk("See p.14 for a complete account.", label='footnote'),
        ]
        processor = make_processor(include_footnotes=True)
        result = processor.process(chunks)
        footnote_results = [r for r in result if r.label == 'footnote']
        assert len(footnote_results) == 1
        assert "p.14" in footnote_results[0].text

    def test_trailing_number_in_footnote_chunk_not_stripped(self) -> None:
        """The existing trailing-number regex must not fire on footnote chunks.
        A footnote ending with a reference like '. 4' should be left intact."""
        chunks = [
            make_chunk("Main body paragraph text."),
            make_chunk("See Jones (1987) for a complete discussion. 4", label='footnote'),
        ]
        processor = make_processor(include_footnotes=True)
        result = processor.process(chunks)
        footnote_results = [r for r in result if r.label == 'footnote']
        assert len(footnote_results) == 1
        assert footnote_results[0].text == "See Jones (1987) for a complete discussion. 4"


class TestStripFootnoteMarkersParameter:
    def test_strip_enabled_by_default_on_body(self) -> None:
        """Body chunks have trailing footnote numbers stripped by default."""
        chunks = [make_chunk("The movement grew rapidly. 4")]
        result = make_processor().process(chunks)
        assert result[0].text == "The movement grew rapidly."

    def test_strip_disabled_leaves_body_unchanged(self) -> None:
        """When strip_footnote_markers=False, body chunks are not stripped."""
        chunks = [make_chunk("The movement grew rapidly. 4")]
        result = make_processor(strip_footnote_markers=False).process(chunks)
        assert result[0].text == "The movement grew rapidly. 4"

    def test_footnote_chunk_protected_even_when_strip_enabled(self) -> None:
        """Footnote chunks are never stripped even when strip_footnote_markers=True."""
        chunks = [
            make_chunk("Body text."),
            make_chunk("See Jones (1987) for details. 4", label='footnote'),
        ]
        result = make_processor(include_footnotes=True, strip_footnote_markers=True).process(chunks)
        footnote_results = [r for r in result if r.label == 'footnote']
        assert footnote_results[0].text == "See Jones (1987) for details. 4"

    def test_formula_chunk_not_stripped_by_footnote_removal(self) -> None:
        """Formula chunks must not have trailing digits stripped as footnote markers.
        Digits at the end of a formula expression (e.g. 'P(A|B).3') are part of the
        math notation, not footnote references."""
        chunks = [make_chunk("P(A|B).3", label='formula')]
        result = make_processor(strip_footnote_markers=True).process(chunks)
        assert result[0].text == "P(A|B).3"


class TestMathSectionHeaderReclassification:
    def test_math_heavy_section_header_reclassified_as_formula(self) -> None:
        """A section header whose text is a math formula should be emitted as a
        formula, not a section header. Docling sometimes misclassifies formula
        lines (e.g. equation '(2) a3/"J\'2 = constant,') as section headers,
        and the processor should catch this."""
        chunks = [make_chunk('(2) a3/"J\'2 = constant,', label='section_header')]
        result = make_processor().process(chunks)
        assert result[0].label == 'formula'
        assert result[0].text == '(2) a3/"J\'2 = constant,'
