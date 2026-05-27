"""Diagnostic script: dump Docling text items from a page with is_small_text detail.

Run from the Book2Audio root:
    python debug/check_page.py

Prints every text item on the target page with bbox info and is_small_text
path analysis so we can see why an item is or isn't being flagged as small text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from docling_core.types.doc.document import DocItemLabel
from utils.docling_utils import (
    load_as_document,
    get_pdf_page_labels,
    compute_single_line_height,
    compute_body_line_height,
    compute_median_chars_per_line,
    is_small_text,
)

PDF = Path("../documents/Realism and the Aim of Science -- Karl Popper -- 2017.pdf")
TARGET_LABEL = "80"

doc = load_as_document(PDF)
page_labels = get_pdf_page_labels(PDF)

target_page_no: int | None = None
for zero_idx, label in page_labels.items():
    if label.lower() == TARGET_LABEL.lower():
        target_page_no = zero_idx + 1
        break

if target_page_no is None:
    print(f"Could not find a page labelled '{TARGET_LABEL}' in the PDF.")
    sys.exit(1)

print(f"Page '{TARGET_LABEL}' is physical page {target_page_no}.\n")

# Compute document-level metrics.
all_text_items = list(doc.texts)
single_line_height = compute_single_line_height(doc)
body_line_height = compute_body_line_height(all_text_items, single_line_height)
median_chars_per_line = compute_median_chars_per_line(all_text_items, single_line_height)

print(f"Document metrics:")
print(f"  single_line_height     = {single_line_height:.3f}")
print(f"  body_line_height       = {body_line_height:.3f}  (75th-pct of single-line TEXT items)")
print(f"  body_line_height * 0.85= {body_line_height * 0.85:.3f}  (Path 1 threshold)")
print(f"  median_chars_per_line  = {median_chars_per_line:.3f}")
print(f"  median * 1.25          = {median_chars_per_line * 1.25:.3f}  (Path 2 threshold)")
print()

# Collect items from the target page.
items = [
    item for item in doc.texts
    if item.prov and item.prov[0].page_no == target_page_no
]

print(f"Found {len(items)} text item(s) on page '{TARGET_LABEL}':\n")
print("-" * 72)

for i, item in enumerate(items, start=1):
    prov = item.prov[0] if item.prov else None
    bbox = prov.bbox if prov else None
    charspan_len = (prov.charspan[1] - prov.charspan[0]) if prov else 0

    small = is_small_text(item, single_line_height, median_chars_per_line,
                          body_line_height=body_line_height)

    # Determine which path fired.
    path = ""
    if small and bbox:
        if body_line_height > 0 and bbox.height < body_line_height * 0.85:
            path = "Path1(bbox.height)"
        else:
            path = "Path2(chars/line)"

    bbox_str = f"bbox.height={bbox.height:.3f}" if bbox else "no bbox"
    chars_per_line = ""
    if bbox and bbox.height > 0 and single_line_height > 0:
        est_lines = bbox.height / single_line_height
        cpl = charspan_len / est_lines if est_lines > 0 else 0
        chars_per_line = f"  chars/line={cpl:.1f}"

    flag = f"  *** SMALL ({path})" if small else ""
    print(f"[{i}] label={item.label}  {bbox_str}  charspan={charspan_len}{chars_per_line}{flag}")
    print(f"     {item.text[:120]}")
    print()
