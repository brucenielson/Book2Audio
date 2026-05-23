"""Tests for AudioGenerator, KokoroEngine, BookToAudio, and load_as_document."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from docling_core.types.doc.document import TextItem, DocItemLabel
from docling_core.types import DoclingDocument
from engines import TTSEngine, KokoroEngine
from audio_generator import AudioGenerator
from book_converter import BookToAudio
from book_to_audio import main as book_to_audio_main
from parsers.docling_parser import DoclingParser
from text_cleaner import FormulaMode
from utils.docling_utils import load_as_document


# --- Helpers for front-matter tests ---

def _make_text_item(text: str, page_no: int) -> MagicMock:
    """Minimal mock TextItem with the given text and page number."""
    item = MagicMock(spec=TextItem)
    item.label = DocItemLabel.TEXT.value
    item.text = text
    prov = MagicMock()
    prov.page_no = page_no
    prov.bbox = MagicMock()
    prov.bbox.height = 10.0
    prov.bbox.t = 0.0
    prov.charspan = (0, len(text))
    item.prov = [prov]
    return item


def _make_real_parser(texts: list, page_labels: dict[int, str],
                      skip_front_matter: bool) -> DoclingParser:
    """Create a DoclingParser backed by a fake DoclingDocument (no file I/O)."""
    doc = MagicMock(spec=DoclingDocument)
    doc.name = "test_doc"
    doc.texts = texts
    doc.pages = {}
    return DoclingParser(
        source=doc,
        include_footnotes=False,
        page_labels=page_labels,
        skip_front_matter=skip_front_matter,
    )


# --- Fixtures ---

@pytest.fixture
def mock_engine():
    """A mock TTSEngine that returns a single audio segment."""
    engine = MagicMock(spec=TTSEngine)
    engine.sample_rate = 24000
    engine.generate.return_value = np.ones(24000, dtype=np.float32)
    return engine


@pytest.fixture
def audio_generator(mock_engine):
    """An AudioGenerator with an injected mock engine."""
    return AudioGenerator(engine=mock_engine)


@pytest.fixture
def mock_audio_generator():
    """A fully mocked AudioGenerator."""
    return MagicMock(spec=AudioGenerator)


@pytest.fixture
def book_to_audio(mock_audio_generator):
    """A BookToAudio instance with an injected mock AudioGenerator."""
    return BookToAudio(audio_generator=mock_audio_generator)


# --- AudioGenerator tests ---

class TestAudioGenerator:
    def test_generate_returns_numpy_array(self, audio_generator) -> None:
        """generate() should return a numpy array."""
        result = audio_generator.generate("Hello world.")
        assert isinstance(result, np.ndarray)

    def test_generate_delegates_to_engine(self, audio_generator, mock_engine) -> None:
        """generate() should delegate to the engine's generate method."""
        audio_generator.generate("Test text.")
        mock_engine.generate.assert_called_once_with("Test text.")

    def test_save_writes_file(self, audio_generator, tmp_path) -> None:
        """save() should write a WAV file to the given path."""
        output_file = str(tmp_path / "output.wav")
        audio = np.zeros(24000, dtype=np.float32)
        audio_generator.save(audio, output_file)
        assert Path(output_file).exists()

    def test_save_uses_engine_sample_rate(self, mock_engine, tmp_path) -> None:
        """save() should use the engine's sample_rate."""
        mock_engine.sample_rate = 48000
        gen = AudioGenerator(engine=mock_engine)
        output_file = str(tmp_path / "output.wav")
        audio = np.zeros(48000, dtype=np.float32)
        with patch('audio_generator.sf.write') as mock_write:
            gen.save(audio, output_file)
            mock_write.assert_called_once_with(output_file, audio, 48000)

    def test_generate_and_save_calls_both(self, audio_generator, tmp_path) -> None:
        """generate_and_save() should produce a WAV file from text."""
        output_file = str(tmp_path / "output.wav")
        audio_generator.generate_and_save("Hello world.", output_file)
        assert Path(output_file).exists()


# --- KokoroEngine tests ---

class TestKokoroEngine:
    def test_generate_returns_numpy_array(self) -> None:
        """generate() should return a numpy array."""
        mock_pipeline = MagicMock()
        audio = np.ones(24000, dtype=np.float32)
        # noinspection SpellCheckingInspection
        mock_pipeline.return_value = [("hello world", "həloʊ wɜrld", audio)]
        engine = KokoroEngine(voice='af_heart', pipeline=mock_pipeline)
        result = engine.generate("Hello world.")
        assert isinstance(result, np.ndarray)

    def test_generate_concatenates_segments(self) -> None:
        """generate() should concatenate multiple audio segments."""
        mock_pipeline = MagicMock()
        segment1 = np.ones(100, dtype=np.float32)
        segment2 = np.ones(200, dtype=np.float32)
        # noinspection SpellCheckingInspection
        mock_pipeline.return_value = [
            ("hello", "həloʊ", segment1),
            ("world", "wɜrld", segment2),
        ]
        engine = KokoroEngine(pipeline=mock_pipeline)
        result = engine.generate("hello world")
        assert len(result) == 300

    def test_generate_calls_pipeline_with_correct_args(self) -> None:
        """generate() should call the pipeline with the correct voice and split pattern."""
        mock_pipeline = MagicMock()
        audio = np.ones(24000, dtype=np.float32)
        # noinspection SpellCheckingInspection
        mock_pipeline.return_value = [("test", "tɛst", audio)]
        engine = KokoroEngine(voice='af_heart', pipeline=mock_pipeline)
        engine.generate("Test text.")
        mock_pipeline.assert_called_once_with(
            "Test text.", voice='af_heart', speed=1.0, split_pattern=r'\n+'
        )

    def test_sample_rate(self) -> None:
        """KokoroEngine should have a sample rate of 24000."""
        mock_pipeline = MagicMock()
        engine = KokoroEngine(pipeline=mock_pipeline)
        assert engine.sample_rate == 24000

    def test_creates_pipeline_if_none(self) -> None:
        """KokoroEngine should create its own pipeline if none is provided."""
        with patch('engines.kokoro.KPipeline') as mock_pipeline_cls:
            with patch('engines.kokoro.torch.cuda.is_available', return_value=False):
                KokoroEngine()
                mock_pipeline_cls.assert_called_once_with(lang_code='a', device='cpu')

    def test_uses_cuda_if_available(self) -> None:
        """KokoroEngine should use CUDA device if available."""
        with patch('engines.kokoro.KPipeline') as mock_pipeline_cls:
            with patch('engines.kokoro.torch.cuda.is_available', return_value=True):
                KokoroEngine()
                mock_pipeline_cls.assert_called_once_with(lang_code='a', device='cuda')


# --- BookToAudio tests ---

class TestBookToAudio:
    def test_text_to_audio_calls_generate_and_save(self, book_to_audio, mock_audio_generator) -> None:
        """convert_to_audio() with a string should generate and save audio."""
        mock_audio_generator.generate.return_value = np.ones(24000, dtype=np.float32)
        book_to_audio.convert_to_audio("Hello world.", "output.wav")
        mock_audio_generator.generate.assert_called_once_with("Hello world.")
        mock_audio_generator.save.assert_called_once_with(mock_audio_generator.generate.return_value, "output.wav")

    def test_document_to_audio_saves_file(self, book_to_audio, mock_audio_generator, tmp_path) -> None:
        """document_to_audio() should generate and save audio for each paragraph."""
        fake_audio = np.ones(24000, dtype=np.float32)
        mock_audio_generator.generate.return_value = fake_audio

        paragraphs = ["First paragraph.", "Second paragraph."]

        with patch('book_converter.DoclingParser') as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.run.return_value = (paragraphs, [])
            mock_parser_cls.return_value = mock_parser

            pdf_path = Path(str(tmp_path / "test_doc.pdf"))
            book_to_audio.convert_to_audio(pdf_path)

            assert mock_audio_generator.generate.call_count == 2
            mock_audio_generator.save.assert_called_once()

    def test_document_to_audio_prints_when_no_paragraphs(self, mock_audio_generator, capsys) -> None:
        """document_to_audio() should print a message if no paragraphs are extracted."""
        converter = BookToAudio(audio_generator=mock_audio_generator, verbose=True)
        with patch('book_converter.DoclingParser') as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.run.return_value = ([], [])
            mock_parser_cls.return_value = mock_parser

            converter.convert_to_audio(Path("test.pdf"))

        captured = capsys.readouterr()
        assert "No paragraphs extracted" in captured.out

    def test_dry_run_skips_audio_generation(self, mock_audio_generator) -> None:
        """document_to_audio() with dry_run should skip audio generation."""
        converter = BookToAudio(audio_generator=mock_audio_generator, dry_run=True)

        with patch('book_converter.DoclingParser') as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.run.return_value = (["A paragraph."], [])
            mock_parser_cls.return_value = mock_parser

            converter.convert_to_audio(Path("test.pdf"))

        mock_audio_generator.generate.assert_not_called()
        mock_audio_generator.save.assert_not_called()

    def test_skip_front_matter_includes_all_pages_by_default(
            self, mock_audio_generator) -> None:
        """With skip_front_matter=False (default), Roman-numbered pages are kept."""
        # Page 1 → label 'i' (front matter), page 2 → label '1' (body)
        texts = [
            _make_text_item("Preface text.", page_no=1),
            _make_text_item("Chapter one body.", page_no=2),
        ]
        parser = _make_real_parser(texts, page_labels={0: 'i', 1: '1'},
                                   skip_front_matter=False)
        converter = BookToAudio(audio_generator=mock_audio_generator, dry_run=True)

        with patch('book_converter.DoclingParser', return_value=parser):
            captured: list[str] = []
            original_run = parser.run

            def capturing_run(**kwargs):
                result = original_run(**kwargs)
                captured.extend(result[0])
                return result

            parser.run = capturing_run
            converter.convert_to_audio(Path("test.pdf"), skip_front_matter=False)

        assert any("Preface" in p for p in captured), "Front matter should be included"
        assert any("Chapter one" in p for p in captured), "Body should be included"

    def test_skip_front_matter_excludes_roman_numeral_pages(
            self, mock_audio_generator) -> None:
        """With skip_front_matter=True, pages i and ii are dropped; page 1 is kept."""
        # Pages 1 and 2 have Roman labels (i, ii); page 3 has Arabic label (1)
        texts = [
            _make_text_item("Front matter page i.", page_no=1),
            _make_text_item("Front matter page ii.", page_no=2),
            _make_text_item("Body text on page 1.", page_no=3),
        ]
        parser = _make_real_parser(texts, page_labels={0: 'i', 1: 'ii', 2: '1'},
                                   skip_front_matter=True)
        converter = BookToAudio(audio_generator=mock_audio_generator, dry_run=True)

        with patch('book_converter.DoclingParser', return_value=parser):
            captured: list[str] = []
            original_run = parser.run

            def capturing_run(**kwargs):
                result = original_run(**kwargs)
                captured.extend(result[0])
                return result

            parser.run = capturing_run
            converter.convert_to_audio(Path("test.pdf"), skip_front_matter=True)

        assert not any("Front matter" in p for p in captured), \
            "Pages i and ii must be excluded"
        assert any("Body text" in p for p in captured), \
            "Arabic-numbered page must be included"


# --- TestFormulaModeThreading ---

class TestFormulaModeThreading:
    """Tests that formula_mode threads correctly from CLI → BookToAudio → parsers."""

    def test_book_to_audio_accepts_formula_mode(self, mock_audio_generator) -> None:
        """BookToAudio.__init__ must accept a formula_mode parameter."""
        converter = BookToAudio(audio_generator=mock_audio_generator,
                                formula_mode=FormulaMode.AUDIO)
        assert converter._formula_mode == FormulaMode.AUDIO

    def test_default_formula_mode_is_skip(self, mock_audio_generator) -> None:
        """BookToAudio defaults to FormulaMode.SKIP when formula_mode is omitted."""
        converter = BookToAudio(audio_generator=mock_audio_generator)
        assert converter._formula_mode == FormulaMode.SKIP

    def test_pdf_parser_receives_formula_mode(self, mock_audio_generator, tmp_path) -> None:
        """BookToAudio passes formula_mode to DoclingParser for PDF files."""
        converter = BookToAudio(audio_generator=mock_audio_generator,
                                formula_mode=FormulaMode.CLEAN, dry_run=True)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        with patch('book_converter.DoclingParser') as mock_parser_cls:
            mock_parser_cls.return_value.run.return_value = ([], [])
            converter.convert_to_audio(pdf_path)
        _, kwargs = mock_parser_cls.call_args
        assert kwargs.get('formula_mode') == FormulaMode.CLEAN

    def test_epub_parser_receives_formula_mode(self, mock_audio_generator, tmp_path) -> None:
        """BookToAudio passes formula_mode to EpubParser for EPUB files."""
        converter = BookToAudio(audio_generator=mock_audio_generator,
                                formula_mode=FormulaMode.AUDIO, dry_run=True)
        epub_path = tmp_path / "test.epub"
        epub_path.touch()
        with patch('book_converter.EpubParser') as mock_parser_cls:
            mock_parser_cls.return_value.run.return_value = ([], [])
            converter.convert_to_audio(epub_path)
        _, kwargs = mock_parser_cls.call_args
        assert kwargs.get('formula_mode') == FormulaMode.AUDIO

    def test_main_default_formula_mode_is_skip(self) -> None:
        """CLI main() passes FormulaMode.SKIP to BookToAudio by default."""
        with patch('book_to_audio.BookToAudio') as mock_cls:
            with patch('book_to_audio._create_engine'):
                book_to_audio_main(text="hello world")
        _, kwargs = mock_cls.call_args
        assert kwargs.get('formula_mode') == FormulaMode.SKIP

    def test_main_passes_formula_mode_audio(self) -> None:
        """CLI main() passes FormulaMode.AUDIO when formula_mode=FormulaMode.AUDIO."""
        with patch('book_to_audio.BookToAudio') as mock_cls:
            with patch('book_to_audio._create_engine'):
                book_to_audio_main(text="hello world", formula_mode=FormulaMode.AUDIO)
        _, kwargs = mock_cls.call_args
        assert kwargs.get('formula_mode') == FormulaMode.AUDIO


# --- load_as_document tests ---

class TestLoadAsDocument:
    def test_loads_from_json_if_exists(self, tmp_path) -> None:
        """load_as_document() should load from JSON cache if it exists."""
        json_path = tmp_path / "test.json"
        json_path.write_text("{}")

        with patch('utils.docling_utils.DoclingDocument.load_from_json') as mock_load:
            mock_load.return_value = MagicMock()
            load_as_document(str(tmp_path / "test.pdf"))
            mock_load.assert_called_once_with(json_path)

    def test_converts_and_saves_if_no_json(self, tmp_path) -> None:
        """load_as_document() should convert the file and save JSON if no cache exists."""
        pdf_path = str(tmp_path / "test.pdf")

        with patch('utils.docling_utils.DocumentConverter') as mock_converter_cls:
            mock_converter = MagicMock()
            mock_converter_cls.return_value = mock_converter
            mock_book = MagicMock()
            mock_converter.convert.return_value.document = mock_book

            load_as_document(pdf_path)

            mock_converter.convert.assert_called_once_with(pdf_path)
            mock_book.save_as_json.assert_called_once()