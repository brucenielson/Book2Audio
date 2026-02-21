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


def simple_generate_and_save_audio(text: str,
                                   output_file: str,
                                   voice: str = 'af_heart',
                                   sample_rate: int = 24000,
                                   play_audio: bool = False):
    """Generate audio from text using Kokoro, play each segment, and save combined audio to a WAV file."""
    pipeline = KPipeline(lang_code='a')
    audio_segments = []

    for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')):
        print(f"Segment {i}: Graphemes: {gs} | Phonemes: {ps}")
        # if play_audio:
        #     sd.play(audio, sample_rate)
        #     sd.wait()
        audio_segments.append(audio)

    combined_audio = np.concatenate(audio_segments)
    sf.write(output_file, combined_audio, sample_rate)
    print(f"Audio saved to {output_file}")


def simple_example():
    text = "Hello, world! This is a test of the Kokoro TTS system."
    simple_generate_and_save_audio(text, "output2.wav", play_audio=True)


def simple_pdf_to_audio(file_path: str):
    if not file_path:
        print("No file path provided.")
        return
    text = load_pdf_text(file_path)
    print("Extracted text from PDF.")
    output_file = get_audio_file_path(file_path)
    simple_generate_and_save_audio(text, output_file=output_file)


def docling_parser_pdf_to_audio(file_path: str,
                                voice: str = 'af_heart',
                                sample_rate: int = 24000):
    # pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    # converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )
    # converter = DocumentConverter()
    # result: ConversionResult = converter.convert(file_path)
    # book: DoclingDocument = result.document
    book = load_pdf_document(file_path)
    valid_pages = load_valid_pages("documents/pdf_valid_pages.csv")
    start_page = None
    end_page = None
    if book.name in valid_pages:
        start_page, end_page = valid_pages[book.name]

    parser = DoclingParser(book, {},
                           min_paragraph_size=300,
                           start_page=start_page,
                           end_page=end_page,
                           double_notes=True)
    paragraphs, meta = parser.run(debug=False)

    if not paragraphs:
        print("No paragraphs extracted from the document.")
        return

    """Generate audio from text using Kokoro, play each segment, and save combined audio to a WAV file."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = KPipeline(lang_code='a', device=device)
    audio_segments = []
    for i, paragraph in enumerate(paragraphs):
        print(f"Generating audio for paragraph {i+1}/{len(paragraphs)}")
        text = paragraph

        for j, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')):
            print(f"Segment {j}: Graphemes: {gs} | Phonemes: {ps}")
            audio_segments.append(audio)

    combined_audio = np.concatenate(audio_segments)
    output_file = get_audio_file_path(file_path)
    sf.write(output_file, combined_audio, sample_rate)
    print(f"Audio saved to {output_file}")


def main(file_path: str = None, use_simple: bool = False):
    if file_path is None:
        file_path = sys.argv[1] if len(sys.argv) > 1 else None

    if use_simple:
        simple_pdf_to_audio(file_path)
    else:
        docling_parser_pdf_to_audio(file_path)

if __name__ == "__main__":
    main(r"documents\The Myth of the Closed Mind.pdf")