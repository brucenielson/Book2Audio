# Run this script with the path to a PDF file as an argument, e.g.:
# python book_to_audio.py "documents/BookTitle.pdf"

from docling_core.types import DoclingDocument
from pathlib import Path
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from docling_parser import DoclingParser
import torch
from typing import Optional, List, Tuple
from docling.document_converter import DocumentConverter
import argparse
# import sounddevice as sd
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.datamodel.base_models import InputFormat


# https://huggingface.co/hexgrad/Kokoro-82M
# https://huggingface.co/hexgrad/Kokoro-82M/discussions/64
# https://huggingface.co/hexgrad/Kokoro-82M/discussions/120
# pip install kokoro
# pip install soundfile


def load_as_document(file_path: str) -> DoclingDocument:
    """Load a document file and return it as a DoclingDocument.

    If a cached JSON file exists at the same path (with a .json extension),
    it will be loaded directly instead of re-converting the source file.
    Otherwise, the file is converted using DocumentConverter and the result
    is saved as JSON for future use.

    Args:
        file_path: Path to the source document file (e.g. a PDF).

    Returns:
        A DoclingDocument representing the parsed document.
    """
    json_path: Path = Path(file_path).with_suffix('.json')
    if json_path.exists():
        return DoclingDocument.load_from_json(json_path)
    converter: DocumentConverter = DocumentConverter()
    result = converter.convert(file_path)
    book: DoclingDocument = result.document
    book.save_as_json(json_path)
    return book


class AudioGenerator:
    """Handles text-to-speech generation and audio file saving.

    Wraps a KPipeline TTS model and provides methods to generate audio
    from text, save audio to disk, or do both in one step.

    Attributes:
        _pipeline: The KPipeline TTS model used for speech synthesis.
        _voice: The voice identifier to use for synthesis (e.g. 'af_heart').
        _sample_rate: The sample rate in Hz for the output audio.
    """

    def __init__(self,
                 pipeline: Optional[KPipeline] = None,
                 voice: str = 'af_heart',
                 sample_rate: int = 24000) -> None:
        """Initialise the AudioGenerator.

        If no pipeline is provided, one will be created automatically,
        using CUDA if available, otherwise falling back to CPU.

        Args:
            pipeline: An optional pre-constructed KPipeline instance.
                      If None, a new pipeline is created automatically.
            voice: The voice identifier to use for TTS synthesis.
                   Defaults to 'af_heart'.
            sample_rate: The sample rate in Hz for the output WAV file.
                         Defaults to 24000.
        """
        if pipeline is None:
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
            pipeline = KPipeline(lang_code='a', device=device)
        self._pipeline: KPipeline = pipeline
        self._voice: str = voice
        self._sample_rate: int = sample_rate

    def generate(self, text: str) -> np.ndarray:
        """Generate audio from a text string and return as a numpy array.

        The text is passed through the TTS pipeline, which may split it
        into multiple segments. All segments are concatenated into a single
        audio array before returning.

        Args:
            text: The text to synthesise into speech.

        Returns:
            A numpy array containing the generated audio samples.
        """
        audio_segments: List[np.ndarray] = []
        for i, (gs, ps, audio) in enumerate(self._pipeline(text, voice=self._voice, speed=1, split_pattern=r'\n+')):
            print(f"Segment {i}: Graphemes: {gs} | Phonemes: {ps}")
            audio_segments.append(audio)
        return np.concatenate(audio_segments)

    def save(self, audio: np.ndarray, output_file: str) -> None:
        """Save a numpy audio array to a WAV file.

        Args:
            audio: A numpy array of audio samples to save.
            output_file: The path to the output WAV file.
        """
        sf.write(output_file, audio, self._sample_rate)
        print(f"Audio saved to {output_file}")

    def generate_and_save(self, text: str, output_file: str) -> None:
        """Generate audio from text and save it directly to a WAV file.

        Convenience method that combines generate() and save() in one call.

        Args:
            text: The text to synthesise into speech.
            output_file: The path to the output WAV file.
        """
        self.save(self.generate(text), output_file)


class BookToAudio:
    """Orchestrates the conversion of text or documents to audio files.

    Composes an AudioGenerator to handle speech synthesis, and provides
    high-level methods for converting raw text or document files to audio.

    Attributes:
        _audio_generator: The AudioGenerator instance used for TTS and saving.
    """

    def __init__(self, audio_generator: Optional[AudioGenerator] = None) -> None:
        """Initialise BookToAudio.

        Args:
            audio_generator: An optional AudioGenerator instance to use.
                             If None, a default AudioGenerator is created.
        """
        self._audio_generator: AudioGenerator = audio_generator or AudioGenerator()

    def text_to_audio(self, text: str, output_file: str) -> None:
        """Generate audio from a text string and save to a WAV file.

        Args:
            text: The text to synthesise into speech.
            output_file: The path to the output WAV file.
        """
        self._audio_generator.generate_and_save(text, output_file)

    def document_to_audio(self, file_path: str,
                          start_page: Optional[int] = None,
                          end_page: Optional[int] = None) -> None:
        """Convert a document to audio using DoclingParser.

        Loads the document, extracts and cleans paragraphs using DoclingParser,
        generates audio for each paragraph, and saves the combined audio as a
        WAV file at the same path as the source file with a .wav extension.

        Args:
            file_path: Path to the source document file (e.g. a PDF).
            start_page: Optional first page to include in the conversion.
                        If None, conversion starts from the beginning.
            end_page: Optional last page to include in the conversion.
                      If None, conversion continues to the end of the document.
        """
        document: DoclingDocument = load_as_document(file_path)
        parser: DoclingParser = DoclingParser(document, {},
                                             min_paragraph_size=300,
                                             start_page=start_page,
                                             end_page=end_page,
                                             double_notes=True)
        paragraphs: List[str]
        paragraphs, _ = parser.run()
        if not paragraphs:
            print("No paragraphs extracted from the document.")
            return
        audio_segments: List[np.ndarray] = []
        for i, paragraph in enumerate(paragraphs):
            print(f"Generating audio for paragraph {i+1}/{len(paragraphs)}")
            audio_segments.append(self._audio_generator.generate(paragraph))
        output_file: str = str(Path(file_path).with_suffix('.wav'))
        self._audio_generator.save(np.concatenate(audio_segments), output_file)


def main(file_path: Optional[str] = None,
         text: Optional[str] = None,
         output_file: Optional[str] = None,
         voice: Optional[str] = None,
         start_page: Optional[int] = None,
         end_page: Optional[int] = None) -> None:
    """Entry point for the book-to-audio conversion tool.

    Parses command line arguments (falling back to the provided parameter
    defaults) and dispatches to either text_to_audio or document_to_audio
    depending on what input is provided.

    Args:
        file_path: Path to the source document file. Can also be provided
                   as the first positional argument on the command line.
        text: A raw text string to convert to audio instead of a file.
        output_file: Path to the output WAV file. Defaults to 'output.wav'
                     when converting text. Ignored for document conversion,
                     which derives the output path from the input file path.
        voice: The TTS voice identifier to use. Defaults to 'af_heart'.
        start_page: Optional first page to include in document conversion.
        end_page: Optional last page to include in document conversion.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('file_path', nargs='?', default=file_path)
    parser.add_argument('--text', default=text)
    parser.add_argument('--output-file', default=output_file or 'output.wav')
    parser.add_argument('--voice', default=voice or 'af_heart')
    parser.add_argument('--start-page', type=int, default=start_page)
    parser.add_argument('--end-page', type=int, default=end_page)
    args: argparse.Namespace = parser.parse_args()

    converter: BookToAudio = BookToAudio(AudioGenerator(voice=args.voice))
    if args.text is not None:
        converter.text_to_audio(args.text, args.output_file)
    elif args.file_path is not None:
        converter.document_to_audio(args.file_path, start_page=args.start_page, end_page=args.end_page)
    else:
        print("No file path or text provided.")


if __name__ == "__main__":
    main(r"documents\The Myth of the Closed Mind.pdf", start_page=129, end_page=129) # 289
