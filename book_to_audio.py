# Run this script with the path to a PDF file as an argument, e.g.:
# python book_to_audio.py "documents/BookTitle.pdf"

from docling_core.types import DoclingDocument
from pathlib import Path
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from docling_parser import DoclingParser
from utils import load_valid_pages
import torch
import sys
from typing import List, Optional
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


def load_pdf_document(file_path: str) -> DoclingDocument:
    """Load a PDF, caching as JSON if needed."""
    json_path = Path(file_path).with_suffix('.json')
    if json_path.exists():
        return DoclingDocument.load_from_json(json_path)
    converter = DocumentConverter()
    result = converter.convert(file_path)
    book = result.document
    book.save_as_json(json_path)
    return book


def load_pdf_text(file_path: str) -> str:
    """Load a PDF, caching as JSON if needed, and export its text."""
    return load_pdf_document(file_path).export_to_text()


class AudioGenerator:
    def __init__(self,
                 pipeline: KPipeline = None,
                 voice: str = 'af_heart',
                 sample_rate: int = 24000):
        if pipeline is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            pipeline = KPipeline(lang_code='a', device=device)
        self._pipeline = pipeline
        self._voice = voice
        self._sample_rate = sample_rate

    def generate(self, text: str) -> np.ndarray:
        """Generate audio from text and return as numpy array."""
        audio_segments = []
        for i, (gs, ps, audio) in enumerate(self._pipeline(text, voice=self._voice, speed=1, split_pattern=r'\n+')):
            print(f"Segment {i}: Graphemes: {gs} | Phonemes: {ps}")
            audio_segments.append(audio)
        return np.concatenate(audio_segments)

    def save(self, audio: np.ndarray, output_file: str):
        """Save a numpy audio array to a WAV file."""
        sf.write(output_file, audio, self._sample_rate)
        print(f"Audio saved to {output_file}")

    def generate_and_save(self, text: str, output_file: str):
        """Generate audio from text and save to a WAV file."""
        self.save(self.generate(text), output_file)


class BookToAudio:
    def __init__(self, audio_generator: AudioGenerator = None):
        self._audio_generator = audio_generator or AudioGenerator()

    def text_to_audio(self, text: str, output_file: str):
        """Generate audio from a text string and save to a WAV file."""
        self._audio_generator.generate_and_save(text, output_file)

    def pdf_to_audio(self, file_path: str,
                     start_page: Optional[int] = None,
                     end_page: Optional[int] = None):
        """Convert a PDF to audio using DoclingParser."""
        document = load_pdf_document(file_path)
        parser = DoclingParser(document, {},
                               min_paragraph_size=300,
                               start_page=start_page,
                               end_page=end_page,
                               double_notes=True)
        paragraphs, _ = parser.run()
        if not paragraphs:
            print("No paragraphs extracted from the document.")
            return
        audio_segments = []
        for i, paragraph in enumerate(paragraphs):
            print(f"Generating audio for paragraph {i+1}/{len(paragraphs)}")
            audio_segments.append(self._audio_generator.generate(paragraph))
        output_file = str(Path(file_path).with_suffix('.wav'))
        self._audio_generator.save(np.concatenate(audio_segments), output_file)


def main(file_path: str = None, text: str = None,
         output_file: str = None,
         voice: str = None,
         start_page: Optional[int] = None, end_page: Optional[int] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', nargs='?', default=file_path)
    parser.add_argument('--text', default=text)
    parser.add_argument('--output-file', default=output_file or 'output.wav')
    parser.add_argument('--voice', default=voice or 'af_heart')
    parser.add_argument('--start-page', type=int, default=start_page)
    parser.add_argument('--end-page', type=int, default=end_page)
    args = parser.parse_args()

    converter = BookToAudio(AudioGenerator(voice=args.voice))
    if args.text is not None:
        converter.text_to_audio(args.text, args.output_file)
    elif args.file_path is not None:
        converter.pdf_to_audio(args.file_path, start_page=args.start_page, end_page=args.end_page)
    else:
        print("No file path or text provided.")


if __name__ == "__main__":
    main(r"documents\The Myth of the Closed Mind.pdf", start_page=129, end_page=129) # 289