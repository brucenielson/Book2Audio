"""Temp script: show exactly what Docling returns for page 367 items, in emission order."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from docling_core.types import DoclingDocument

TARGET_PAGE = 293
path = "documents/Realism and the Aim of Science -- Karl Popper -- 2017.json"

doc = DoclingDocument.model_validate_json(open(path, encoding='utf-8').read())

items = [
    item for item in doc.texts
    if item.prov and item.prov[0].page_no == TARGET_PAGE
]

print(f"Page {TARGET_PAGE}: {len(items)} items in Docling emission order\n")
print(f"{'#':<4} {'label':<18} {'t':>7} {'b':>7} {'l':>7} {'r':>7}  {'origin':<12}  text")
print("-" * 110)

for i, item in enumerate(items, 1):
    prov = item.prov[0]
    bbox = prov.bbox
    if bbox is not None:
        t  = f"{bbox.t:.2f}"
        b  = f"{bbox.b:.2f}"
        l  = f"{bbox.l:.2f}"
        r  = f"{bbox.r:.2f}"
        origin = bbox.coord_origin.value if hasattr(bbox.coord_origin, 'value') else str(bbox.coord_origin)
    else:
        t = b = l = r = "None"
        origin = "None"

    text = (item.text or "").replace('\n', ' ')
    print(f"{i:<4} {str(item.label.value):<18} {t:>7} {b:>7} {l:>7} {r:>7}  {origin:<12}")
    print(f"     {repr(text)}")
    print()
