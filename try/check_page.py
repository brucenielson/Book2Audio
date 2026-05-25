"""Diagnostic script: dump all Docling raw chunks from page xxix of the Popper PDF.

Run from the Book2Audio root:
    python check_page.py

Prints every text item Docling found on that page, numbered, with its label
and repr()-quoted text so trailing characters are unambiguous.
"""

import sys
from pathlib import Path

# Make sure the project root is on the path so local imports work.
sys.path.insert(0, str(Path(__file__).parent))

from utils.docling_utils import load_as_document, get_pdf_page_labels

PDF = Path("../documents/Realism and the Aim of Science -- Karl Popper -- 2017.pdf")
TARGET_LABEL = "xxix"

doc = load_as_document(PDF)
page_labels = get_pdf_page_labels(PDF)   # {0-based-index -> label-string}

# Find the physical (1-based) page number that carries the printed label "xxix".
target_page_no: int | None = None
for zero_idx, label in page_labels.items():
    if label.lower() == TARGET_LABEL.lower():
        target_page_no = zero_idx + 1   # Docling page_no is 1-based
        break

if target_page_no is None:
    print(f"Could not find a page labelled '{TARGET_LABEL}' in the PDF.")
    sys.exit(1)

print(f"Page '{TARGET_LABEL}' is physical page {target_page_no}.\n")

# Collect all text items from that page in Docling's emission order.
items = [
    item
    for item in doc.texts
    if item.prov and item.prov[0].page_no == target_page_no
]

print(f"Found {len(items)} text item(s) on page {TARGET_LABEL}:\n")
print("-" * 72)

for i, item in enumerate(items, start=1):
    print(f"[{i}] label={item.label}")
    print(f"     {item.text}")
    print()
