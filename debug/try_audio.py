"""Produce audio from any supported file (PDF, EPUB, TXT) with LLM cleaning.

Simple debug script for testing audio output end-to-end. Edit the variables
at the top to point at your document and tune the options.

Usage: run via "Run Current File" from the debug/ directory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from book_to_audio import main

# --- Configure here ---

FILE = Path(r"..\documents\A Sceptical Theory of Scientific Inquiry 2.pdf")

OUTPUT_FILE = None  # None → auto-named alongside the source file

# ----------------------------------------------------------------------

if not FILE.exists():
    print(f"ERROR: File not found: {FILE.resolve()}")
    sys.exit(1)

main(
    file_path=str(FILE),
    output_file=OUTPUT_FILE,
    dry_run=False,
    generate_text_file=True,
    # llm_cleaner=True,
    # llm_model='llama3.1:8b',
    formula_mode='none',
    verbose=True,
    show_pages=True,
    skip_index=True,
    skip_front_matter=True,
    include_footnotes=False,
    end_page=222,
)
