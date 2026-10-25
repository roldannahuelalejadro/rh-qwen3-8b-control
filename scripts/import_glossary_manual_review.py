"""Validate and import the human labels without changing acquired responses."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/glossary"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    content = args.source.read_bytes()
    reviews = json.loads(content)
    expected = {f"r{i:04d}" for i in range(1, 101)}
    if set(reviews) != expected:
        raise ValueError("Expected exactly the 100 rollout identifiers.")
    for key, review in reviews.items():
        response = ROOT / "data/glossary_2026-10-23" / key / "result.json"
        if review["y"] not in ("0", "1", "X") or not review["revisor"].strip():
            raise ValueError(f"Invalid label/reviewer: {key}")
        if hashlib.sha256(response.read_bytes()).hexdigest() != review["result_sha256"]:
            raise ValueError(f"Response hash mismatch: {key}")
        datetime.fromisoformat(review["utc"].replace("Z", "+00:00"))
    OUT.mkdir(exist_ok=True)
    original = OUT / "revision_manual_Nahuel_2026-10-24_original.json"
    if original.exists() and original.read_bytes() != content:
        raise ValueError("The original review archive already exists with different contents.")
    active = OUT / "revision_manual.json"
    if active.exists() and json.loads(active.read_text(encoding="utf-8")) != reviews:
        raise ValueError("An active revision differs; reconcile before replacing it.")
    if not original.exists():
        shutil.copyfile(args.source, original)
    if not active.exists():
        shutil.copyfile(args.source, active)
    print("Imported 100 human labels; response hashes verified; original review preserved.")


if __name__ == "__main__":
    main()
