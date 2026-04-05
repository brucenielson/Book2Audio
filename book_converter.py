from pathlib import Path
from typing import List
import numpy as np

from audio_generator import AudioGenerator
from parsers.docling_parser import DoclingParser
from parsers.epub_parser import EpubParser


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

    def convert_to_audio(self, source: str | Path,
                         output_file: str | None = None,
                         start_page: int | None = None,
                         end_page: int | None = None,
                         generate_text_file: bool = False,
                         sections_to_skip: List[str] | None = None) -> None:
        """Convert text, a PDF, an EPUB, or a TXT file to audio.

        Dispatches to the appropriate parser based on the type and extension
        of the source. A plain string is converted directly. A Path is
        inspected for its extension and routed to the correct parser.

        Args:
            source: Either a raw text string to convert, or a Path to a
                    .pdf, .epub, or .txt file.
            output_file: Optional path to the output WAV file. If None, the
                         output file is derived from the source path with a
                         .wav extension. Ignored for raw text conversion.
            start_page: Optional first page to include. PDF only.
            end_page: Optional last page to include. PDF only.
            generate_text_file: If True, saves processed text and paragraph
                                files alongside the source document.
            sections_to_skip: Optional list of EPUB section IDs to skip in
                              addition to any sections listed in the CSV file.
                              EPUB only.
        """
        paragraphs: List[str]

        if isinstance(source, str):
            # Raw text string — convert directly
            if not self._dry_run:
                audio: np.ndarray = self._audio_generator.generate(source)
                self._audio_generator.save(audio, output_file or 'output.wav')
            else:
                print("Dry run: Did not generate audio.")
            return

        # File path — dispatch by extension
        suffix: str = source.suffix.lower()
        if suffix == '.txt':
            paragraphs = [source.read_text(encoding='utf-8')]
        elif suffix == '.pdf':
            parser: DoclingParser = DoclingParser(source, include_footnotes=False, start_page=start_page, end_page=end_page)
            paragraphs, _ = parser.run(generate_text_file=generate_text_file)
        elif suffix == '.epub':
            epub_parser: EpubParser = EpubParser(source, include_footnotes=False, sections_to_skip=sections_to_skip)
            paragraphs, _ = epub_parser.run(generate_text_file=generate_text_file)
        else:
            raise ValueError(f"Unsupported file type: '{suffix}'. Supported types: .pdf, .epub, .txt")

        if self._dry_run:
            print("Dry run: Did not generate audio.")
            return
        elif not paragraphs:
            print("No paragraphs extracted from the document.")
            return

        audio_segments: List[np.ndarray] = []
        for i, paragraph in enumerate(paragraphs):
            print(f"Generating audio for paragraph {i + 1}/{len(paragraphs)}")
            audio_segments.append(self._audio_generator.generate(paragraph))
        output_file = output_file or str(source.with_suffix('.wav'))
        self._audio_generator.save(np.concatenate(audio_segments), output_file)
