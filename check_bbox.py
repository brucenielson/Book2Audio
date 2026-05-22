import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

path = "documents/Realism and the Aim of Science -- Karl Popper -- 2017.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = data.get("texts", [])

# Get the full text (not truncated) for items near the problem area
for target_page in [367]:
    items_on_page = [
        (item, prov)
        for item in texts
        for prov in item.get("prov", [])
        if prov.get("page_no") == target_page
    ]

    # Sort by -t as code does
    items_on_page.sort(key=lambda x: -(x[1].get("bbox") or {}).get("t", 0))

    print(f"\n=== Page {target_page} SORTED, full text ===")
    for i, (item, prov) in enumerate(items_on_page):
        bbox = prov.get("bbox") or {}
        t = bbox.get("t", 0)
        label = item.get("label", "?")
        text = item.get("text", "").encode('ascii', errors='replace').decode()
        first_char = repr(text[0]) if text else "EMPTY"
        print(f"\n  [{i+1}] t={t:.1f}  label={label}  first={first_char}")
        print(f"       {repr(text[:120])}")
