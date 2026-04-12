import pytest
from unittest.mock import MagicMock, patch
from text_chunk import RawChunk, ParsedChunk
from text_processor import TextProcessor, _all_words_valid


# --- Fixtures ---

def make_chunk(text: str, label: str = 'text', page: str = '') -> RawChunk:
    """Create a RawChunk with the given text, label, and optional page number."""
    meta = {'page_#': page} if page else {}
    return RawChunk(text=text, meta=meta, label=label)


def make_processor(min_paragraph_size: int = 0,
                   include_footnotes: bool = False) -> TextProcessor:
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
    def test_empty_chunks_returns_empty(self):
        processor = make_processor()
        result = processor.process([])
        assert result == []

    def test_single_complete_sentence(self):
        processor = make_processor()
        chunks = [make_chunk("This is a complete sentence.")]
        result = processor.process(chunks)
        assert len(result) == 1
        assert result[0].text == "This is a complete sentence."

    def test_incomplete_sentence_accumulated_with_next(self):
        processor = make_processor()
        chunks = [
            make_chunk("This is incomplete"),
            make_chunk("and this completes it."),
        ]
        result = processor.process(chunks)
        assert len(result) == 1
        assert "This is incomplete" in result[0].text
        assert "and this completes it." in result[0].text

    def test_incomplete_sentence_at_end_still_emitted(self):
        processor = make_processor()
        chunks = [make_chunk("This is incomplete")]
        result = processor.process(chunks)
        assert len(result) == 1
        assert "This is incomplete" in result[0].text

    def test_short_paragraphs_accumulated_until_min_size(self):
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

    def test_section_header_emitted_as_own_paragraph(self):
        processor = make_processor()
        chunks = [make_chunk("Chapter One", label='section_header')]
        result = processor.process(chunks)
        assert len(result) == 1
        assert result[0].text == "Chapter One"

    def test_section_header_flushes_accumulated_paragraph(self):
        processor = make_processor(min_paragraph_size=100)
        chunks = [
            make_chunk("Accumulated text."),
            make_chunk("Chapter One", label='section_header'),
        ]
        result = processor.process(chunks)
        assert any("Accumulated text." in r.text for r in result)
        assert any("Chapter One" in r.text for r in result)

    def test_section_header_resets_accumulator(self):
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

    def test_section_name_in_meta_after_header(self):
        processor = make_processor()
        chunks = [
            make_chunk("Chapter One", label='section_header'),
            make_chunk("Content here."),
        ]
        result = processor.process(chunks)
        content = next(r for r in result if "Content here." in r.text)
        assert content.meta["section_name"] == "Chapter One"

    def test_page_header_skipped(self):
        processor = make_processor()
        chunks = [
            make_chunk("Page Header", label='page_header'),
            make_chunk("Real content."),
        ]
        result = processor.process(chunks)
        assert all("Page Header" not in r.text for r in result)

    def test_page_footer_skipped(self):
        processor = make_processor()
        chunks = [
            make_chunk("Real content."),
            make_chunk("Page Footer", label='page_footer'),
        ]
        result = processor.process(chunks)
        assert all("Page Footer" not in r.text for r in result)

    def test_footnote_excluded_by_default(self):
        processor = make_processor(include_footnotes=False)
        chunks = [
            make_chunk("Main text."),
            make_chunk("Footnote text.", label='footnote'),
        ]
        result = processor.process(chunks)
        assert all("Footnote text." not in r.text for r in result)

    def test_footnote_included_when_flag_set(self):
        processor = make_processor(include_footnotes=True)
        chunks = [
            make_chunk("Main text."),
            make_chunk("Footnote text.", label='footnote'),
        ]
        result = processor.process(chunks)
        assert any("Footnote text." in r.text for r in result)

    def test_paragraph_numbers_increment(self):
        processor = make_processor()
        chunks = [
            make_chunk("First paragraph."),
            make_chunk("Second paragraph."),
        ]
        result = processor.process(chunks)
        assert result[0].meta["paragraph_#"] == "1"
        assert result[1].meta["paragraph_#"] == "2"

    def test_meta_passed_through(self):
        processor = make_processor()
        chunk = RawChunk(text="Some content.", meta={"source": "test"}, label='text')
        result = processor.process([chunk])
        assert result[0].meta["source"] == "test"

    def test_next_section_header_forces_emit(self):
        processor = make_processor(min_paragraph_size=1000)
        chunks = [
            make_chunk("Short paragraph."),
            make_chunk("Chapter Two", label='section_header'),
        ]
        result = processor.process(chunks)
        assert any("Short paragraph." in r.text for r in result)

    def test_generate_text_file_creates_file(self, tmp_path):
        processor = make_processor()
        chunks = [make_chunk("Some content.")]
        output_path = tmp_path / "test"
        processor.process(chunks, output_path=output_path, generate_text_file=True)
        assert (tmp_path / "test_processed_paragraphs.txt").exists()

    def test_generate_text_file_content(self, tmp_path):
        processor = make_processor()
        chunks = [make_chunk("Some content.")]
        output_path = tmp_path / "test"
        processor.process(chunks, output_path=output_path, generate_text_file=True)
        content = (tmp_path / "test_processed_paragraphs.txt").read_text(encoding="utf-8")
        assert "Some content." in content

    def test_generate_text_file_false_does_not_create_file(self, tmp_path):
        processor = make_processor()
        chunks = [make_chunk("Some content.")]
        output_path = tmp_path / "test"
        processor.process(chunks, output_path=output_path, generate_text_file=False)
        assert not (tmp_path / "test_processed_paragraphs.txt").exists()

    def test_process_can_be_called_multiple_times(self):
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
    def test_cleaner_body_paragraph_kept(self):
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk("Some b0dy text.")])  # digit forces cleaner call
        assert len(result) == 1
        assert "Some b0dy text." in result[0].text

    def test_cleaner_drop_paragraph_discarded(self):
        cleaner = make_cleaner(classification='drop')
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk("Table of contents ... 1")])  # digit forces cleaner call
        assert result == []

    def test_cleaner_footnote_excluded_by_default(self):
        cleaner = make_cleaner(classification='footnote')
        processor = TextProcessor(cleaner=cleaner, include_footnotes=False)
        result = processor.process([make_chunk("1 A footnote.")])  # digit forces cleaner call
        assert result == []

    def test_cleaner_footnote_included_when_flag_set(self):
        cleaner = make_cleaner(classification='footnote', cleaned="A footnote.")
        processor = TextProcessor(cleaner=cleaner, include_footnotes=True)
        result = processor.process([make_chunk("1 A footnote.")])  # digit forces cleaner call
        assert len(result) == 1
        assert result[0].label == 'footnote'

    def test_cleaner_cleaned_text_used(self):
        cleaner = make_cleaner(classification='body', cleaned="Fixed text.")
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk("Brok en text.")])  # misspelling forces cleaner call
        assert result[0].text == "Fixed text."

    def test_cleaner_receives_page_context(self):
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        chunks = [
            make_chunk("F1rst paragraph.", page='1'),   # digit forces cleaner call
            make_chunk("S3cond paragraph.", page='1'),  # digit forces cleaner call
        ]
        processor.process(chunks)
        call_args = cleaner.clean.call_args
        assert call_args[1]['page_context'] != '' or call_args[0][1] != ''

    def test_cleaner_called_once_per_flushed_paragraph(self):
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        chunks = [make_chunk("F1rst."), make_chunk("S3cond.")]  # digits force cleaner calls
        processor.process(chunks)
        assert cleaner.clean.call_count == 2

    def test_no_cleaner_uses_existing_behavior(self):
        processor = make_processor()
        chunks = [make_chunk("Normal text.")]
        result = processor.process(chunks)
        assert len(result) == 1
        assert result[0].text == "Normal text."


# --- TestAllWordsValid ---

class TestAllWordsValid:
    def test_all_valid_words_returns_true(self):
        """A sentence of common English words should return True."""
        assert _all_words_valid("The dog ran quickly") is True

    def test_ocr_artifact_returns_false(self):
        """A sentence containing a non-word should return False."""
        assert _all_words_valid("I am hppy today") is False

    def test_standalone_number_returns_false(self):
        """A standalone numeric token is a potential artifact and should return False."""
        assert _all_words_valid("Chapter 1789") is False

    def test_embedded_digit_returns_false(self):
        """A digit embedded in a word (e.g. OCR artifact) should return False."""
        assert _all_words_valid("The dog ran quickly1.") is False

    def test_empty_string_returns_true(self):
        """An empty string has no invalid words, so _all_words_valid returns True."""
        assert _all_words_valid("") is True

    def test_punctuation_stripped_before_check(self):
        """Punctuation attached to valid words should be ignored."""
        assert _all_words_valid("Hello, world.") is True


# --- TestSkipCleanerWhenAllWordsValid ---

class TestSkipCleanerWhenAllWordsValid:
    def test_cleaner_not_called_when_all_words_valid(self):
        """Cleaner should not be called when every word in the paragraph is valid."""
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk("The dog ran quickly.")])
        cleaner.clean.assert_not_called()

    def test_paragraph_returned_unchanged_when_all_words_valid(self):
        """Paragraph should pass through unmodified when all words are valid."""
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        result = processor.process([make_chunk("The dog ran quickly.")])
        assert result[0].text == "The dog ran quickly."

    def test_cleaner_called_when_invalid_word_present(self):
        """Cleaner should be called when the paragraph contains an invalid word."""
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk("The dog ran qukckly.")])
        cleaner.clean.assert_called_once()

    def test_cleaner_called_when_digit_present(self):
        """Cleaner should be called when the paragraph contains a digit token."""
        cleaner = make_cleaner(classification='body')
        processor = TextProcessor(cleaner=cleaner)
        processor.process([make_chunk("See footnote 4 for details.")])
        cleaner.clean.assert_called_once()
