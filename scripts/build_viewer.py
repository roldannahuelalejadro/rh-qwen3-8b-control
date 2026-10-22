"""Regenerate the offline viewer using existing records, without inference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/control_2026-10-21"
TITLES = ["Frecuencias de palabras", "Duplicados consecutivos", "Hamlet y frase de verificación",
          "Glosario de lingüística", "Reseña de Gatsby", "Diálogo y turnos", "Omelette y chocolate",
          "Helado napolitano", "Evaluador rápido A", "Evaluador rápido B"]
STAGES = ("original_nf4", "original_bf16", "sft_nf4", "sft_bf16")
FIELDS = ("prompt_id", "family", "checkpoint", "precision", "rollout_index", "text",
          "seed", "output_tokens", "truncated", "finish_reason", "scoring", "created_at")


def main():
    protocol = json.loads((DATA / "protocol.json").read_text(encoding="utf-8"))
    prompts = [{**p, "title": title} for p, title in zip(protocol["prompts"], TITLES)]
    records = []
    for stage in STAGES:
        for path in sorted((DATA / stage / "rollouts").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            records.append({key: row[key] for key in FIELDS})
    keys = {(r["prompt_id"], r["checkpoint"], r["precision"], r["rollout_index"]) for r in records}
    if len(keys) != 400 or len(records) != 400 or len(prompts) != 10:
        raise ValueError("Incomplete experiment; refusing to generate viewer")
    payload = json.dumps({"prompts": prompts, "records": records}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = (ROOT / "viewer/template.html").read_text(encoding="utf-8")
    (ROOT / "viewer/index.html").write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print("viewer/index.html: 10 prompts, 400 responses, no model calls")


if __name__ == "__main__":
    main()
