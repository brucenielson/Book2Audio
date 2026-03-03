# Book to Audio

Converts PDF books and text files to audio using text-to-speech (TTS). Parses PDFs into clean, semantically meaningful paragraphs and synthesises them into a WAV file.

---

## Requirements

- Python 3.10+
- GPU recommended for faster TTS (CUDA-compatible)
- Dependencies: `docling`, `kokoro`, `nltk`, `torch`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Command Line

```bash
python book_to_audio.py <file_path> [options]
```

**Examples:**

Convert a PDF to audio:
```bash
python book_to_audio.py documents/my_book.pdf
```

Convert specific pages:
```bash
python book_to_audio.py documents/my_book.pdf --start-page 10 --end-page 50
```

Convert a plain text file:
```bash
python book_to_audio.py documents/my_text.txt
```

Convert a raw text string:
```bash
python book_to_audio.py --text "Hello, this is a test."
```

Dry run (parse only, no audio generated):
```bash
python book_to_audio.py documents/my_book.pdf --dry-run
```

Generate debug text files alongside the PDF:
```bash
python book_to_audio.py documents/my_book.pdf --generate-text-file
```

### Options

| Option | Description | Default |
|---|---|---|
| `file_path` | Path to the PDF or TXT file to convert | — |
| `--text` | Raw text string to convert instead of a file | — |
| `--output-file` | Path to the output WAV file | `output.wav` |
| `--voice` | TTS voice identifier | `af_heart` |
| `--start-page` | First page to include (PDF only) | — |
| `--end-page` | Last page to include (PDF only) | — |
| `--dry-run` | Parse the document but skip audio generation | `False` |
| `--generate-text-file` | Save processed text and paragraph files alongside the source document | `False` |

---

## Python API

### Convert a PDF

```python
from book_to_audio import BookToAudio, AudioGenerator

converter = BookToAudio(AudioGenerator(voice='af_heart'))
converter.document_to_audio('documents/my_book.pdf', start_page=10, end_page=50)
```

### Convert text

```python
converter.text_to_audio("Some text to convert.", output_file='output.wav')
```

### Dry run with debug files

```python
converter = BookToAudio(AudioGenerator(), dry_run=True)
converter.document_to_audio('documents/my_book.pdf', generate_text_file=True)
```

---

## Debug Files

When `--generate-text-file` is used, two files are written to the same folder as the source PDF:

- `<name>_processed_texts.txt` — every DocItem extracted from the PDF, one per line, with page number and label
- `<name>_processed_paragraphs.txt` — the final chunked paragraphs that would be passed to TTS

These are useful for inspecting how the parser is handling a particular document before committing to a full audio conversion.

---

## PDF Caching

On first conversion, the parsed PDF is saved as a JSON file alongside the source. Subsequent runs load from this cache, skipping the slow PDF conversion step.

---

## Project Structure

```
book_to_audio.py        # Main entry point and BookToAudio/AudioGenerator classes
docling_parser.py       # DoclingParser — PDF parsing and paragraph chunking
word_validator.py       # WordValidator — hyphen resolution using NLTK
utils/
    docling_utils.py    # Helper functions for text cleaning and DocItem inspection
tests/
    test_book_to_audio.py
    test_docling_parser.py
    test_docling_utils.py
    test_word_validator.py
    test_document_output.py
    test_documents/     # Input PDFs for integration tests
    test_canonical/     # Expected outputs for integration tests
```
