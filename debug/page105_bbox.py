"""Debug script: print bbox.t positions for all text items on page 105 (physical page 146)
of Realism and the Aim of Science, relative to median_page_height.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.docling_utils import load_as_document, compute_median_page_height, compute_single_line_height, compute_body_line_height, compute_median_chars_per_line, is_small_text

DOC_PATH = Path(__file__).parent.parent / "tests" / "test_documents" / "Realism and the Aim of Science -- Karl Popper -- 2017.pdf"
TARGET_PAGE = 146  # physical (Docling) page number

doc = load_as_document(DOC_PATH)

median_page_height = compute_median_page_height(doc)
single_line_height = compute_single_line_height(doc)

all_text_items = [item for item in doc.texts if item.prov and item.prov[0].bbox is not None]
body_line_height = compute_body_line_height(all_text_items, single_line_height)
median_chars_per_line = compute_median_chars_per_line(all_text_items, single_line_height)

print(f"median_page_height : {median_page_height:.1f}")
print(f"single_line_height : {single_line_height:.1f}")
print(f"body_line_height   : {body_line_height:.1f}")
print(f"median_chars/line  : {median_chars_per_line:.1f}")
print(f"50% threshold      : {median_page_height * 0.5:.1f}")
print()
print(f"{'bbox.t':>8}  {'%page':>6}  {'half':>6}  {'small?':>6}  {'bbox.h':>6}  {'est_ln':>6}  {'ch/ln':>6}  {'thresh':>6}  text[:50]")
print("-" * 120)

threshold = 1.25
for item in all_text_items:
    prov = item.prov[0]
    if prov.page_no != TARGET_PAGE:
        continue
    bbox_t = prov.bbox.t
    bbox_h = prov.bbox.height
    pct = (bbox_t / median_page_height * 100) if median_page_height > 0 else 0
    half = "lower" if bbox_t < median_page_height * 0.5 else "upper"
    small = is_small_text(item, single_line_height, median_chars_per_line, body_line_height=body_line_height)
    charspan = prov.charspan[1] - prov.charspan[0] if prov.charspan else 0
    est_lines = bbox_h / single_line_height if single_line_height > 0 else 0
    chars_per_line = charspan / est_lines if est_lines > 0 else 0
    thresh = median_chars_per_line * threshold
    text_preview = (item.text or "")[:50].replace('\n', ' ')
    print(f"{bbox_t:>8.1f}  {pct:>5.1f}%  {half:>6}  {str(small):>6}  {bbox_h:>6.1f}  {est_lines:>6.1f}  {chars_per_line:>6.1f}  {thresh:>6.1f}  {text_preview}")
