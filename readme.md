# Book to Audio

Converts PDF books, EPUB files, and text files to audio using text-to-speech (TTS). Parses documents into clean, semantically meaningful paragraphs and synthesises them into a WAV file.

---

## Requirements

- Python 3.10+
- GPU recommended for faster TTS (CUDA-compatible)
- Dependencies: `docling`, `kokoro`, `nltk`, `torch`, `ebooklib`, `beautifulsoup4`

Install dependencies:

    pip install -r requirements.txt

---

## Usage

### Command Line

    python book_to_audio.py <file_path> [options]

**Examples:**

Convert a PDF to audio:

    python book_to_audio.py documents/my_book.pdf

Convert specific pages:

    python book_to_audio.py documents/my_book.pdf --start-page 10 --end-page 50

Convert an EPUB to audio:

    python book_to_audio.py documents/my_book.epub

Convert an EPUB, skipping front matter sections:

    python book_to_audio.py documents/my_book.epub --sections-to-skip cover titlepage toc

Convert a plain text file:

    python book_to_audio.py documents/my_text.txt

Convert a raw text string:

    python book_to_audio.py --text "Hello, this is a test."

Dry run (parse only, no audio generated):

    python book_to_audio.py documents/my_book.pdf --dry-run

Generate debug text files alongside the source document:

    python book_to_audio.py documents/my_book.pdf --generate-text-file

### Options

- `file_path` — path to the PDF, EPUB, or TXT file to convert
- `--text` — raw text string to convert instead of a file
- `--output-file` — path to the output WAV file (default: `output.wav`)
- `--voice` — Kokoro voice identifier (default: `af_heart`)
- `--engine` — TTS engine to use: `kokoro` or `qwen` (default: `kokoro`)
- `--speaker` — Qwen speaker name (default: `vivian`)
- `--language` — Qwen language, e.g. `English`, `Auto` (default: `Auto`)
- `--instruct` — Qwen style instruction, e.g. `speak calmly` (default: none)
- `--model-size` — Qwen model size: `0.6b` or `1.7b` (default: `0.6b`)
- `--start-page` — first page to include, PDF only
- `--end-page` — last page to include, PDF only
- `--sections-to-skip` — one or more EPUB section IDs to skip, separated by spaces
- `--dry-run` — parse the document but skip audio generation
- `--generate-text-file` — save processed text and paragraph files alongside the source document

---

## Python API

### Convert a PDF

    from book_converter import BookToAudio
    from audio_generator import AudioGenerator

    converter = BookToAudio(AudioGenerator())
    converter.convert_to_audio(Path('documents/my_book.pdf'), start_page=10, end_page=50)

### Convert an EPUB

    converter.convert_to_audio(Path('documents/my_book.epub'),
                               sections_to_skip=['cover', 'titlepage', 'toc'])

### Convert text

    converter.convert_to_audio("Some text to convert.", output_file='output.wav')

### Dry run with debug files

    converter = BookToAudio(AudioGenerator(), dry_run=True)
    converter.convert_to_audio(Path('documents/my_book.pdf'), generate_text_file=True)

---

## Debug Files

When `--generate-text-file` is used, files are written to the same folder as the source document:

**PDF:**
- `<name>_processed_texts.txt` — every DocItem extracted from the PDF, one per line, with page number and label
- `<name>_processed_paragraphs.txt` — the final chunked paragraphs that would be passed to TTS

**EPUB:**
- `<name>_processed_paragraphs.txt` — the final chunked paragraphs
- `<name>_processed_meta.txt` — metadata alongside each paragraph, useful for verifying chapter and section attribution

These are useful for inspecting how the parser is handling a particular document before committing to a full audio conversion. Run with `--dry-run --generate-text-file` first, inspect the output, then convert.

---

## Skipping EPUB Sections

To find section IDs to skip, run with `--generate-text-file` and inspect `_processed_meta.txt` — the `item_id` field shows the section ID for each paragraph. Pass unwanted IDs to `--sections-to-skip`.

If you want to load sections to skip from a CSV file, use `load_sections_to_skip` from `utils/general_utils.py` and pass the result to `EpubParser` directly:

    from utils.general_utils import load_sections_to_skip
    from parsers.epub_parser import EpubParser

    sections = load_sections_to_skip(Path('sections_to_skip.csv'))
    parser = EpubParser('my_book.epub', meta_data={},
                        sections_to_skip=list(sections.get('My Book Title', [])))

---

## PDF Caching

On first conversion, the parsed PDF is saved as a JSON file alongside the source. Subsequent runs load from this cache, skipping the slow PDF conversion step.

---

## Project Structure

    book_to_audio.py        # CLI entry point
    book_converter.py       # BookToAudio — orchestrates conversion
    audio_generator.py      # AudioGenerator — TTS and WAV output
    engines.py              # KokoroEngine, QwenCustomVoiceEngine
    word_validator.py       # WordValidator — hyphen resolution using NLTK
    text_chunk.py           # RawChunk, ParsedChunk dataclasses
    text_processor.py       # TextProcessor — paragraph accumulation
    parsers/
        docling_parser.py   # DoclingParser — PDF parsing
        epub_parser.py      # EpubParser — EPUB parsing
    utils/
        general_utils.py    # Shared text cleaning and utility functions
        docling_utils.py    # DocItem inspection helpers and PDF-specific cleaning
    tests/
        test_book_to_audio.py
        test_docling_parser.py
        test_epub_parser.py
        test_text_processor.py
        test_docling_utils.py
        test_general_utils.py
        test_word_validator.py
        test_document_output.py
        test_documents/     # Input documents for integration tests
        test_canonical/     # Expected outputs for integration tests