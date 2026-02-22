import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from book_to_audio import AudioGenerator, BookToAudio, load_as_document


# --- Fixtures ---

@pytest.fixture
def mock_pipeline():
    """A mock KPipeline that returns a single audio segment."""
    pipeline = MagicMock()
    audio = np.ones(24000, dtype=np.float32)
    pipeline.return_value = [("hello world", "həloʊ wɜrld", audio)]
    return pipeline


@pytest.fixture
def audio_generator(mock_pipeline):
    """An AudioGenerator with an injected mock pipeline."""
    return AudioGenerator(pipeline=mock_pipeline, voice='af_heart', sample_rate=24000)


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
    def test_generate_returns_numpy_array(self, audio_generator):
        """generate() should return a numpy array."""
        result = audio_generator.generate("Hello world.")
        assert isinstance(result, np.ndarray)

    def test_generate_concatenates_segments(self, mock_pipeline):
        """generate() should concatenate multiple audio segments."""
        segment1 = np.ones(100, dtype=np.float32)
        segment2 = np.ones(200, dtype=np.float32)
        mock_pipeline.return_value = [
            ("hello", "həloʊ", segment1),
            ("world", "wɜrld", segment2),
        ]
        generator = AudioGenerator(pipeline=mock_pipeline)
        result = generator.generate("hello world")
        assert len(result) == 300

    def test_generate_calls_pipeline_with_correct_args(self, audio_generator, mock_pipeline):
        """generate() should call the pipeline with the correct voice and split pattern."""
        audio_generator.generate("Test text.")
        mock_pipeline.assert_called_once_with(
            "Test text.", voice='af_heart', speed=1, split_pattern=r'\n+'
        )

    def test_save_writes_file(self, audio_generator, tmp_path):
        """save() should write a WAV file to the given path."""
        output_file = str(tmp_path / "output.wav")
        audio = np.zeros(24000, dtype=np.float32)
        audio_generator.save(audio, output_file)
        assert Path(output_file).exists()

    def test_generate_and_save_calls_both(self, audio_generator, tmp_path):
        """generate_and_save() should produce a WAV file from text."""
        output_file = str(tmp_path / "output.wav")
        audio_generator.generate_and_save("Hello world.", output_file)
        assert Path(output_file).exists()

    def test_creates_pipeline_if_none(self):
        """AudioGenerator should create its own pipeline if none is provided."""
        with patch('book_to_audio.KPipeline') as mock_kpipeline:
            with patch('book_to_audio.torch.cuda.is_available', return_value=False):
                generator = AudioGenerator()
                mock_kpipeline.assert_called_once_with(lang_code='a', device='cpu')

    def test_uses_cuda_if_available(self):
        """AudioGenerator should use CUDA device if available."""
        with patch('book_to_audio.KPipeline') as mock_kpipeline:
            with patch('book_to_audio.torch.cuda.is_available', return_value=True):
                generator = AudioGenerator()
                mock_kpipeline.assert_called_once_with(lang_code='a', device='cuda')


# --- BookToAudio tests ---

class TestBookToAudio:
    def test_text_to_audio_calls_generate_and_save(self, book_to_audio, mock_audio_generator):
        """text_to_audio() should delegate to generate_and_save on the AudioGenerator."""
        book_to_audio.text_to_audio("Hello world.", "output.wav")
        mock_audio_generator.generate_and_save.assert_called_once_with("Hello world.", "output.wav")

    def test_document_to_audio_saves_file(self, book_to_audio, mock_audio_generator, tmp_path):
        """document_to_audio() should generate and save audio for each paragraph."""
        fake_audio = np.ones(24000, dtype=np.float32)
        mock_audio_generator.generate.return_value = fake_audio

        fake_doc = MagicMock()
        fake_doc.name = "test_doc"
        paragraphs = ["First paragraph.", "Second paragraph."]

        with patch('book_to_audio.load_as_document', return_value=fake_doc):
            with patch('book_to_audio.DoclingParser') as mock_parser_cls:
                mock_parser = MagicMock()
                mock_parser.run.return_value = (paragraphs, [])
                mock_parser_cls.return_value = mock_parser

                pdf_path = str(tmp_path / "test_doc.pdf")
                book_to_audio.document_to_audio(pdf_path)

                assert mock_audio_generator.generate.call_count == 2
                mock_audio_generator.save.assert_called_once()

    def test_document_to_audio_prints_when_no_paragraphs(self, book_to_audio, capsys):
        """document_to_audio() should print a message if no paragraphs are extracted."""
        fake_doc = MagicMock()

        with patch('book_to_audio.load_as_document', return_value=fake_doc):
            with patch('book_to_audio.DoclingParser') as mock_parser_cls:
                mock_parser = MagicMock()
                mock_parser.run.return_value = ([], [])
                mock_parser_cls.return_value = mock_parser

                book_to_audio.document_to_audio("test.pdf")

        captured = capsys.readouterr()
        assert "No paragraphs extracted" in captured.out

    def test_creates_default_audio_generator_if_none(self):
        """BookToAudio should create a default AudioGenerator if none is provided."""
        with patch('book_to_audio.AudioGenerator') as mock_gen_cls:
            BookToAudio()
            mock_gen_cls.assert_called_once()


# --- load_as_document tests ---

class TestLoadAsDocument:
    def test_loads_from_json_if_exists(self, tmp_path):
        """load_as_document() should load from JSON cache if it exists."""
        json_path = tmp_path / "test.json"
        json_path.write_text("{}")  # minimal placeholder

        with patch('book_to_audio.DoclingDocument.load_from_json') as mock_load:
            mock_load.return_value = MagicMock()
            load_as_document(str(tmp_path / "test.pdf"))
            mock_load.assert_called_once_with(json_path)

    def test_converts_and_saves_if_no_json(self, tmp_path):
        """load_as_document() should convert the file and save JSON if no cache exists."""
        pdf_path = str(tmp_path / "test.pdf")

        with patch('book_to_audio.DocumentConverter') as mock_converter_cls:
            mock_converter = MagicMock()
            mock_converter_cls.return_value = mock_converter
            mock_book = MagicMock()
            mock_converter.convert.return_value.document = mock_book

            load_as_document(pdf_path)

            mock_converter.convert.assert_called_once_with(pdf_path)
            mock_book.save_as_json.assert_called_once()
