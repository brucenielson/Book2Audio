# Run this script with the path to a PDF file as an argument, e.g.:
# python book_to_audio.py "documents/BookTitle.pdf"
# python book_to_audio.py "documents/BookTitle.pdf" --engine qwen --speaker vivian --language English
# python book_to_audio.py --text "Hello world" --engine qwen --speaker ryan
from pathlib import Path
from typing import List
import argparse

from engines import KokoroEngine, QwenCustomVoiceEngine, QWEN_SPEAKERS
from audio_generator import AudioGenerator
from book_converter import BookToAudio


def _create_engine(args: argparse.Namespace):
    """Create the appropriate TTS engine based on CLI arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        A TTSEngine instance (KokoroEngine or QwenCustomVoiceEngine).
    """
    if args.engine == 'qwen':
        return QwenCustomVoiceEngine(
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            model_size=args.model_size,
        )
    else:
        return KokoroEngine(voice=args.voice)


def main(file_path: str | None = None,
         text: str | None = None,
         output_file: str | None = None,
         voice: str | None = None,
         start_page: str | int | None = None,
         end_page: str | int | None = None,
         dry_run: bool = False,
         generate_text_file: bool = False,
         engine: str | None = None,
         speaker: str | None = None,
         language: str | None = None,
         instruct: str | None = None,
         model_size: str | None = None,
         sections_to_skip: List[str] | None = None) -> None:
    """Entry point for the book-to-audio conversion tool.

    Parses command line arguments (falling back to the provided parameter
    defaults) and dispatches to either text_to_audio or document_to_audio
    depending on what input is provided.

    Args:
        file_path: Path to the source document file.
        text: A raw text string to convert to audio instead of a file.
        output_file: Path to the output WAV file.
        voice: The Kokoro voice identifier. Defaults to 'af_heart'.
        start_page: Optional first page for document conversion.
        end_page: Optional last page for document conversion.
        dry_run: If True, processes the document but skips audio generation.
        generate_text_file: If True, saves processed text files alongside the source.
        engine: TTS engine to use: 'kokoro' or 'qwen'. Defaults to 'kokoro'.
        speaker: Qwen speaker name (e.g. 'vivian'). Defaults to 'vivian'.
        language: Qwen language (e.g. 'English', 'Auto'). Defaults to 'Auto'.
        instruct: Qwen style instruction (e.g. 'speak calmly'). Defaults to None.
        model_size: Qwen model size: '0.6b' or '1.7b'. Defaults to '0.6b'.
        sections_to_skip: Optional list of EPUB section IDs to skip in addition
                          to any sections listed in the CSV file.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Convert text or documents to audio using Kokoro or Qwen3-TTS.')

    # General arguments
    parser.add_argument('file_path', nargs='?', default=file_path)
    parser.add_argument('--text', default=text)
    parser.add_argument('--output-file', default=output_file or 'output.wav')
    parser.add_argument('--start-page', type=int, default=start_page)
    parser.add_argument('--end-page', type=int, default=end_page)
    parser.add_argument('--dry-run', action='store_true', default=dry_run)
    parser.add_argument('--generate-text-file', action='store_true', default=generate_text_file)
    parser.add_argument('--sections-to-skip', nargs='*', default=sections_to_skip)

    # Engine selection
    parser.add_argument('--engine', choices=['kokoro', 'qwen'], default=engine or 'kokoro',
                        help='TTS engine to use (default: kokoro)')

    # Kokoro-specific arguments
    parser.add_argument('--voice', default=voice or 'af_heart',
                        help='Kokoro voice identifier (default: af_heart)')

    # Qwen-specific arguments
    parser.add_argument('--speaker', default=speaker or 'vivian', choices=QWEN_SPEAKERS,
                        help='Qwen speaker name (default: vivian)')
    parser.add_argument('--language', default=language or 'Auto',
                        help='Qwen language: Auto, English, Chinese, Japanese, etc. (default: Auto)')
    parser.add_argument('--instruct', default=instruct,
                        help='Qwen style instruction, e.g. "speak calmly" (default: none)')
    parser.add_argument('--model-size', default=model_size or '0.6b', choices=['0.6b', '1.7b'],
                        help='Qwen model size (default: 0.6b)')

    args: argparse.Namespace = parser.parse_args()

    engine = _create_engine(args)
    audio_gen: AudioGenerator = AudioGenerator(engine)
    converter: BookToAudio = BookToAudio(audio_gen, dry_run=args.dry_run)

    supported_file_types: List[str] = ['.pdf', '.epub', '.txt']
    if args.text is not None:
        converter.convert_to_audio(args.text, args.output_file)
    elif args.file_path is None:
        raise ValueError("No file path or text provided.")
    elif not Path(args.file_path).exists():
        raise FileNotFoundError(f"File not found: {args.file_path}")
    elif not Path(args.file_path).is_file():
        raise ValueError(f"Path is not a file: {args.file_path}")
    elif Path(args.file_path).suffix.lower() not in supported_file_types:
        raise ValueError(
            f"Unsupported file type: '{Path(args.file_path).suffix}'. Supported types: {supported_file_types}")
    else:
        converter.convert_to_audio(Path(args.file_path), start_page=args.start_page, end_page=args.end_page,
                                   generate_text_file=args.generate_text_file,
                                   sections_to_skip=args.sections_to_skip)


if __name__ == "__main__":
    main(r"documents\The Myth of the Closed Mind.pdf",
         start_page=129, end_page=129, dry_run=False, generate_text_file=True)
    # main(r"documents\Realism and the Aim of Science -- Karl Popper -- 2017.pdf",
    #      start_page=None, end_page=None, dry_run=True, generate_text_file=True)
    # main(r"documents\The Declaration of Independence.epub",
    #      dry_run=True,
    #      generate_text_file=True,
    #      sections_to_skip=["pg-footer", "ncx"])
