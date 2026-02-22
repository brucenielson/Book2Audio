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
from docling.document_converter import DocumentConverter
# from docling.document_converter import DocumentConverter, ConversionResult
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


def get_audio_file_path(pdf_file_path: str) -> str:
    return pdf_file_path.replace('.pdf', '.wav')


class BookToAudio:
    def __init__(self,
                 voice: str = 'af_heart',
                 sample_rate: int = 24000):
        self.voice = voice
        self.sample_rate = sample_rate
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = KPipeline(lang_code='a', device=device)

    def _generate_audio(self, text: str) -> np.ndarray:
        """Generate audio from text and return as numpy array."""
        audio_segments = []
        for i, (gs, ps, audio) in enumerate(self.pipeline(text, voice=self.voice, speed=1, split_pattern=r'\n+')):
            print(f"Segment {i}: Graphemes: {gs} | Phonemes: {ps}")
            audio_segments.append(audio)
        return np.concatenate(audio_segments)

    def save_audio(self, audio: np.ndarray, output_file: str):
        """Save a numpy audio array to a WAV file."""
        sf.write(output_file, audio, self.sample_rate)
        print(f"Audio saved to {output_file}")

    def text_to_audio(self, text: str, output_file: str):
        """Generate audio from a text string and save to a WAV file."""
        self.save_audio(self._generate_audio(text), output_file)

    def pdf_to_audio(self, file_path: str):
        """Convert a PDF to audio using DoclingParser."""
        book = load_pdf_document(file_path)
        valid_pages = load_valid_pages("documents/pdf_valid_pages.csv")
        start_page, end_page = valid_pages.get(book.name, (None, None))
        parser = DoclingParser(book, {},
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
            audio_segments.append(self._generate_audio(paragraph))
        self.save_audio(np.concatenate(audio_segments), get_audio_file_path(file_path))


def main(file_path: str = None, text: str = None):
    if file_path is None:
        file_path = sys.argv[1] if len(sys.argv) > 1 else None
    converter = BookToAudio()
    if text is not None:
        converter.text_to_audio(text, "output.wav")
    elif file_path is not None:
        converter.pdf_to_audio(file_path)
    else:
        print("No file path or text provided.")


if __name__ == "__main__":
    main(r"documents\The Myth of the Closed Mind.pdf")