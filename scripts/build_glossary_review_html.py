"""Build a standalone browser for all 100 glossary responses."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/glossary"
rows = list(csv.DictReader((OUT / "clasificacion_100.csv").open(encoding="utf-8-sig")))
assert len(rows) == 100
manual_path = OUT / "revision_manual.json"
if manual_path.exists():
    manual = json.loads(manual_path.read_text(encoding="utf-8"))
    for row in rows:
        if row["rollout"] in manual:
            row["revision_manual"] = manual[row["rollout"]]
protocol = json.loads((ROOT / "data/glossary_2026-10-23/protocol.json").read_text(encoding="utf-8"))
template = (ROOT / "scripts/glossary_review_template.html").read_text(encoding="utf-8")
review_js = (ROOT / "scripts/glossary_manual_review.js").read_text(encoding="utf-8")
template = template.replace("__REVIEW_JS__", review_js)
payload = json.dumps({"rows": rows, "prompt": protocol["prompt"]}, ensure_ascii=True).replace("<", "\\u003c")
target = OUT / "explorador_glosario_100.html"
target.write_text(template.replace("__DATA__", payload), encoding="utf-8")
saved = target.read_text(encoding="utf-8")
embedded = saved.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
assert json.loads(embedded)["rows"] == rows
assert sum(r["y_propuesto"] == "X" for r in rows) == 33
print(target)
print(f"{len(rows)} respuestas incorporadas; {target.stat().st_size:,} bytes; sin dependencias externas.")
