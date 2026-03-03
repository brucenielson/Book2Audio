# Run this script with the path to a PDF file as an argument, e.g.:
# python book_to_audio.py "documents/BookTitle.pdf"
from pathlib import Path
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from docling_parser import DoclingParser
import torch
from typing import List
import argparse
# import sounddevice as sd
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.datamodel.base_models import InputFormat


# https://huggingface.co/hexgrad/Kokoro-82M
# https://huggingface.co/hexgrad/Kokoro-82M/discussions/64
# https://huggingface.co/hexgrad/Kokoro-82M/discussions/120
# pip install kokoro
# pip install soundfile


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
                 pipeline: KPipeline | None = None,
                 voice: str = 'af_heart',
                 sample_rate: int = 24000) -> None:
        """Initialize the AudioGenerator.

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
            text: The text to synthesize into speech.

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
            text: The text to synthesize into speech.
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

    def __init__(self, audio_generator: AudioGenerator | None = None, dry_run: bool = False) -> None:
        """Initialise BookToAudio.

        Args:
            audio_generator: An optional AudioGenerator instance to use.
                             If None, a default AudioGenerator is created.
        """
        self._audio_generator: AudioGenerator = audio_generator or AudioGenerator()
        self._dry_run: bool = dry_run

    def text_to_audio(self, text: str, output_file: str) -> None:
        """Generate audio from a text string and save to a WAV file.

        Args:
            text: The text to synthesize into speech.
            output_file: The path to the output WAV file.
        """
        self._audio_generator.generate_and_save(text, output_file)

    def document_to_audio(self, file_path: str,
                          start_page: int | None = None,
                          end_page: int | None = None,
                          output_file: str | None = None,
                          generate_text_file: bool = False) -> None:
        """Convert a document to audio using DoclingParser.

        Loads the document, extracts and cleans paragraphs using DoclingParser,
        generates audio for each paragraph, and saves the combined audio as a
        WAV file.

        Args:
            file_path: Path to the source document file (e.g. a PDF).
            start_page: Optional first page to include in the conversion.
                        If None, conversion starts from the beginning.
            end_page: Optional last page to include in the conversion.
                      If None, conversion continues to the end of the document.
            output_file: Optional path to the output WAV file. If None, the
                         output file is derived from file_path with a .wav extension.
            generate_text_file: If True, saves processed text and paragraph files
        """
        parser: DoclingParser = DoclingParser(file_path, {},
                                             min_paragraph_size=300,
                                             start_page=start_page,
                                             end_page=end_page,
                                             include_notes=False)
        paragraphs: List[str]
        paragraphs, _ = parser.run(generate_text_file=generate_text_file)
        if self._dry_run:
            print(f"Dry run: Did not generate audio.")
            return
        elif not paragraphs:
            print("No paragraphs extracted from the document.")
            return

        audio_segments: List[np.ndarray] = []
        for i, paragraph in enumerate(paragraphs):
            print(f"Generating audio for paragraph {i+1}/{len(paragraphs)}")
            audio_segments.append(self._audio_generator.generate(paragraph))
        output_file = output_file or str(Path(file_path).with_suffix('.wav'))
        self._audio_generator.save(np.concatenate(audio_segments), output_file)


def main(file_path: str | None = None,
         text: str | None = None,
         output_file: str | None = None,
         voice: str | None = None,
         start_page: str | int | None = None,
         end_page: str | int | None = None,
         dry_run: bool = False,
         generate_text_file: bool = False) -> None:
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
        dry_run: If True, processes the document but skips audio generation.
        generate_text_file: If True, saves processed text and paragraph files alongside the source document.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('file_path', nargs='?', default=file_path)
    parser.add_argument('--text', default=text)
    parser.add_argument('--output-file', default=output_file or 'output.wav')
    parser.add_argument('--voice', default=voice or 'af_heart')
    parser.add_argument('--start-page', type=int, default=start_page)
    parser.add_argument('--end-page', type=int, default=end_page)
    parser.add_argument('--dry-run', action='store_true', default=dry_run)
    parser.add_argument('--generate-text-file', action='store_true', default=generate_text_file)
    args: argparse.Namespace = parser.parse_args()

    converter: BookToAudio = BookToAudio(AudioGenerator(voice=args.voice), dry_run=args.dry_run)

    supported_file_types: List[str] = ['.pdf', '.txt']
    if args.text is not None:
        converter.text_to_audio(args.text, args.output_file)
    elif args.file_path is None:
        raise ValueError("No file path or text provided.")
    elif not Path(args.file_path).exists():
        raise FileNotFoundError(f"File not found: {args.file_path}")
    elif not Path(args.file_path).is_file():
        raise ValueError(f"Path is not a file: {args.file_path}")
    elif Path(args.file_path).suffix.lower() not in supported_file_types:
        raise ValueError(
            f"Unsupported file type: '{Path(args.file_path).suffix}'. Supported types: {supported_file_types}")
    elif Path(args.file_path).suffix.lower() == '.txt':
        converter.text_to_audio(Path(args.file_path).read_text(encoding='utf-8'), args.output_file)
    else:
        converter.document_to_audio(args.file_path, start_page=args.start_page, end_page=args.end_page,
                                    generate_text_file=args.generate_text_file)


if __name__ == "__main__":
    # main(r"documents\The Myth of the Closed Mind.pdf",
    #      start_page=129, end_page=289, dry_run=True, generate_text_file=True)
    main(r"documents\A World of Propensities -- Karl Popper -- 2018.pdf",
         start_page=None, end_page=None, dry_run=True, generate_text_file=True)
