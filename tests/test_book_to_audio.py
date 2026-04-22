"""Tests for AudioGenerator, KokoroEngine, BookToAudio, and load_as_document."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
from engines import TTSEngine, KokoroEngine
from audio_generator import AudioGenerator
from book_converter import BookToAudio
from utils.docling_utils import load_as_document


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
        mock_pipeline.return_value = [("hello world", "həloʊ wɜrld", audio)]
        engine = KokoroEngine(voice='af_heart', pipeline=mock_pipeline)
        result = engine.generate("Hello world.")
        assert isinstance(result, np.ndarray)

    def test_generate_concatenates_segments(self) -> None:
        """generate() should concatenate multiple audio segments."""
        mock_pipeline = MagicMock()
        segment1 = np.ones(100, dtype=np.float32)
        segment2 = np.ones(200, dtype=np.float32)
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
        with patch('engines.kokoro.KPipeline') as mock_kpipeline:
            with patch('engines.kokoro.torch.cuda.is_available', return_value=False):
                engine = KokoroEngine()
                mock_kpipeline.assert_called_once_with(lang_code='a', device='cpu')

    def test_uses_cuda_if_available(self) -> None:
        """KokoroEngine should use CUDA device if available."""
        with patch('engines.kokoro.KPipeline') as mock_kpipeline:
            with patch('engines.kokoro.torch.cuda.is_available', return_value=True):
                engine = KokoroEngine()
                mock_kpipeline.assert_called_once_with(lang_code='a', device='cuda')


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

    def test_document_to_audio_prints_when_no_paragraphs(self, book_to_audio, capsys) -> None:
        """document_to_audio() should print a message if no paragraphs are extracted."""
        with patch('book_converter.DoclingParser') as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.run.return_value = ([], [])
            mock_parser_cls.return_value = mock_parser

            book_to_audio.convert_to_audio(Path("test.pdf"))

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