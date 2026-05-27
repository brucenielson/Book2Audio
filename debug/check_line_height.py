"""Show how single_line_height and body_line_height are calculated for the Popper PDF.

single_line_height: median bbox.height of PAGE_HEADER and PAGE_FOOTER items.
body_line_height:   75th-percentile bbox.height of single-line body TEXT items.

Run from the Book2Audio root:
    python debug/check_line_height.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from docling_core.types.doc.document import DocItemLabel
from utils.docling_utils import (
    load_as_document,
    compute_single_line_height,
    compute_body_line_height,
    is_page_header,
    is_page_footer,
)

PDF = Path("../documents/Realism and the Aim of Science -- Karl Popper -- 2017.pdf")

doc = load_as_document(PDF)
all_items = list(doc.texts)

single_line_height = compute_single_line_height(doc)
body_line_height = compute_body_line_height(all_items, single_line_height)

print(f"single_line_height      = {single_line_height:.3f}  (median of page header/footer bboxes)")
print(f"body_line_height        = {body_line_height:.3f}  (75th-pct of single-line TEXT bboxes)")
print(f"body_line_height * 0.85 = {body_line_height * 0.85:.3f}  (Path 1 threshold for is_small_text)")
print()

# Show the raw header/footer heights that feed single_line_height.
hf_heights = sorted(
    item.prov[0].bbox.height
    for item in all_items
    if (is_page_header(item) or is_page_footer(item))
    and item.prov and item.prov[0].bbox is not None
)
print(f"Page header/footer bbox heights ({len(hf_heights)} items):")
print(f"  min={hf_heights[0]:.3f}  median={hf_heights[len(hf_heights)//2]:.3f}  max={hf_heights[-1]:.3f}")
print(f"  values: {[round(h, 2) for h in hf_heights[:20]]}{'...' if len(hf_heights) > 20 else ''}")
