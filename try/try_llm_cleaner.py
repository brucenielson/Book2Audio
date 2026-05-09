"""Run the LLM cleaner on a PDF and snapshot the output for comparison.

Runs book_to_audio with the LLM cleaner enabled (dry run, no audio),
then copies the processed paragraphs file into output/ ready to commit
in GitHub Desktop.

Usage: run via "Run Current File" from the try/ directory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
from pathlib import Path
from book_to_audio import main
from utils.general_utils import extract_pdf_pages


# DOC_NAME = r"Realism and the Aim of Science - Intro Test - 2017"
DOC_NAME = r"Realism and the Aim of Science -- Karl Popper -- 2017"

# Set to a (start, end) tuple of physical page numbers to extract a subset of
# the PDF before running. Physical pages are 1-indexed from the front of the
# file — roman-numeral intro pages are still pages 1, 2, 3, … Set to None to
# run on the full document.
EXTRACT_PAGES: tuple[int, int] | None = None
# EXTRACT_PAGES = (12, 22)

PDF = Path(r"..\documents") / (DOC_NAME + ".pdf")
SOURCE = Path(r"..\documents\\" + DOC_NAME + "_processed_paragraphs.txt")
DEST = Path(r"..\output") / SOURCE.name

if EXTRACT_PAGES is not None:
    extracted_name = f"{DOC_NAME} -- extracted {EXTRACT_PAGES[0]} to {EXTRACT_PAGES[1]}"
    PDF = extract_pdf_pages(PDF, PDF.parent / (extracted_name + ".pdf"),
                            EXTRACT_PAGES[0], EXTRACT_PAGES[1])
    SOURCE = PDF.parent / (extracted_name + "_processed_paragraphs.txt")
    DEST = Path(r"..\output") / SOURCE.name

if not PDF.exists():
    print(f"ERROR: PDF not found: {PDF.resolve()}")
    sys.exit(1)
if not DEST.parent.exists():
    print(f"ERROR: Output directory not found: {DEST.parent.resolve()}")
    sys.exit(1)

main(
    file_path=str(PDF),
    dry_run=True,
    generate_text_file=True,
    llm_cleaner=True,
    llm_model='llama3.1:8b',
    verbose=True,
)

if not SOURCE.exists():
    print(f"ERROR: Output file not found: {SOURCE}")
    sys.exit(1)

shutil.copy(SOURCE, DEST)
print(f"\nSaved to {DEST}")
print("Commit in GitHub Desktop to record this run.")
