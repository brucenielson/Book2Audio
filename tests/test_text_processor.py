"""Tests for the TextProcessor class and _all_words_valid helper."""

from unittest.mock import MagicMock
from text_chunk import RawChunk
from text_processor import TextProcessor, _all_words_valid


# --- Fixtures ---

def make_chunk(text: str, label: str = 'text', page: str = '') -> RawChunk:
    """Create a RawChunk with the given text, label, and optional page number."""
    meta = {'page_#': page} if page else {}
    return RawChunk(text=text, meta=meta, label=label)


def make_processor(min_paragraph_size: int = 0,
                   include_footnotes: bool = False) -> TextProcessor:
    """Create a TextProcessor with the given settings."""
    return TextProcessor(min_paragraph_size=min_paragraph_size,
                         include_footnotes=include_footnotes)


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
        assert result[0].label == 'footnote'

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


# --- TestAllWordsValid ---

class TestAllWordsValid:
    def test_all_valid_words_returns_true(self) -> None:
        """A sentence of common English words should return True."""
        assert _all_words_valid("The dog ran quickly") is True

    def test_ocr_artifact_returns_false(self) -> None:
        """A sentence containing a non-word should return False."""
        assert _all_words_valid("I am hppy today") is False

    def test_standalone_number_returns_false(self) -> None:
        """A standalone numeric token is a potential artifact and should return False."""
        assert _all_words_valid("Chapter 1789") is False

    def test_embedded_digit_returns_false(self) -> None:
        """A digit embedded in a word (e.g. OCR artifact) should return False."""
        assert _all_words_valid("The dog ran quickly1.") is False

    def test_empty_string_returns_true(self) -> None:
        """An empty string has no invalid words, so _all_words_valid returns True."""
        assert _all_words_valid("") is True

    def test_punctuation_stripped_before_check(self) -> None:
        """Punctuation attached to valid words should be ignored."""
        assert _all_words_valid("Hello, world.") is True


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
