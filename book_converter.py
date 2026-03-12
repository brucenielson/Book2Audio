from pathlib import Path
from typing import List
import numpy as np

from audio_generator import AudioGenerator
from docling_parser import DoclingParser


class BookToAudio:
    """Orchestrates the conversion of text or documents to audio files.

    Composes an AudioGenerator to handle speech synthesis, and provides
    high-level methods for converting raw text or document files to audio.

    Attributes:
        _audio_generator: The AudioGenerator instance used for TTS and saving.
        _dry_run: If True, processes the document but skips audio generation.
    """

    def __init__(self, audio_generator: AudioGenerator, dry_run: bool = False) -> None:
        """Initialise BookToAudio.

        Args:
            audio_generator: The AudioGenerator instance to use.
            dry_run: If True, processes the document but skips audio generation.
        """
        self._audio_generator: AudioGenerator = audio_generator
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
            print("Dry run: Did not generate audio.")
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
