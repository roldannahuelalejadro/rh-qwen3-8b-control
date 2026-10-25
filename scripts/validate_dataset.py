"""Read-only validation of hashes, records, activation arrays, and viewer data."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/control_2026-10-21"
MANIFEST = ROOT / "MANIFEST.sha256.json"
STAGES = ("original_nf4", "original_bf16", "sft_nf4", "sft_bf16")
KEYS = [f"l{layer}_{summary}" for layer in (6, 18, 30) for summary in ("mean", "last", "tail16_mean")]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def data_path(value):
    path = (DATA / value.replace("\\", "/")).resolve()
    require(path.is_relative_to(DATA.resolve()), "Record points outside the dataset")
    return path


def check_array(array, shape):
    require(array.shape == shape, f"Wrong array shape: {array.shape}, expected {shape}")
    require(np.isfinite(array).all(), "Non-finite activation array")


def inventory():
    return sorted(path for folder in (DATA, ROOT / "acquisition/frozen_source")
                  for path in folder.rglob("*") if path.is_file())


def validate(initialize=False):
    files = inventory()
    if initialize:
        require(not MANIFEST.exists(), "Manifest already exists; refusing to replace the baseline")
    else:
        manifest = read(MANIFEST)
        require(set(manifest["files"]) == {p.relative_to(ROOT).as_posix() for p in files}, "Manifest inventory mismatch")
        for relative, value in manifest["files"].items():
            path = ROOT / relative
            require(path.stat().st_size == value["bytes"] and sha(path) == value["sha256"], f"Integrity mismatch: {relative}")

    protocol = read(DATA / "protocol.json")
    signed = dict(protocol)
    protocol_hash = signed.pop("protocol_sha256")
    require(hashlib.sha256(json.dumps(signed, sort_keys=True, ensure_ascii=True).encode()).hexdigest() == protocol_hash, "Protocol digest mismatch")
    for name, checksum in protocol["code_sha256"].items():
        require(sha(ROOT / "acquisition/frozen_source" / name) == checksum, f"Frozen code mismatch: {name}")
    require(len(protocol["prompts"]) == 10, "Expected 10 prompts")
    records, initial_count, trajectory_count, calibration_count = {}, 0, 0, 0
    for stage in STAGES:
        checkpoint, precision = stage.split("_")
        basis_file = DATA / f"{checkpoint}_nf4/basis.npz"
        basis_hash = sha(basis_file)
        with np.load(basis_file, allow_pickle=False) as basis:
            require(set(basis.files) == {key+suffix for key in KEYS for suffix in ("_center", "_components", "_explained_variance_ratio")}, "Basis keys mismatch")
            for key in KEYS:
                check_array(basis[key+"_center"], (4096,))
                check_array(basis[key+"_components"], (6, 4096))
                check_array(basis[key+"_explained_variance_ratio"], (6,))
            calibration = list((DATA / stage / "calibration").glob("*.npz"))
            require(len(calibration) == 24, f"Calibration coverage mismatch: {stage}")
            for path in calibration:
                with np.load(path, allow_pickle=False) as arrays:
                    require(set(arrays.files) == set(KEYS), "Calibration keys mismatch")
                    for key in KEYS:
                        check_array(arrays[key], (4096,))
                calibration_count += 1
            for prompt in protocol["prompts"]:
                pid = prompt["id"]
                initial_file = DATA / stage / "initials" / (pid+".npz")
                metadata = read(initial_file.with_suffix(".json"))
                require(metadata["protocol_sha256"] == protocol_hash and metadata["basis_sha256"] == basis_hash, "Initial metadata mismatch")
                with np.load(initial_file, allow_pickle=False) as arrays:
                    require(set(arrays.files) == set(KEYS + [key+"_xi" for key in KEYS]), "Initial keys mismatch")
                    for key in KEYS:
                        check_array(arrays[key], (4096,))
                        check_array(arrays[key+"_xi"], (6,))
                        require(arrays[key].dtype == np.float32, "Expected float32 initial vectors")
                        projected = (arrays[key] - basis[key+"_center"]) @ basis[key+"_components"].T
                        require(np.allclose(projected, arrays[key+"_xi"], rtol=1e-5, atol=2e-4), "Stored PCA coordinates mismatch")
                    xi_start = arrays["l18_last_xi"].copy()
                initial_count += 1
                for rollout in range(1, 11):
                    path = DATA / stage / "rollouts" / f"{pid}__r{rollout:02d}.json"
                    row = read(path)
                    key = (pid, checkpoint, precision, rollout-1)
                    require((row["prompt_id"], row["checkpoint"], row["precision"], row["rollout_index"]) == key, "Record identity mismatch")
                    require(row["protocol_sha256"] == protocol_hash and row["basis_sha256"] == basis_hash, "Record protocol/basis mismatch")
                    require(row["input_ids_sha256"] == metadata["input_ids_sha256"], "Input identity mismatch")
                    require(data_path(row["initial_file"]) == initial_file.resolve(), "Wrong initial reference")
                    trajectory = data_path(row["trajectory_file"])
                    with np.load(trajectory, allow_pickle=False) as arrays:
                        require(set(arrays.files) == {"generated_tokens_processed", "processed_sequence_position", "xi_last"}, "Trajectory keys mismatch")
                        expected_steps = np.arange(0, row["output_tokens"], 8)
                        require(np.array_equal(arrays["generated_tokens_processed"], expected_steps), "Trajectory step mismatch")
                        require(np.array_equal(arrays["processed_sequence_position"], expected_steps+row["input_tokens"]-1), "Sequence positions mismatch")
                        check_array(arrays["xi_last"], (len(expected_steps), 6))
                        require(np.allclose(arrays["xi_last"][0], xi_start, rtol=1e-5, atol=2e-4), "Trajectory initial point mismatch")
                    records[key] = row
                    trajectory_count += 1

    html = (ROOT / "viewer/index.html").read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="data">([\s\S]*?)</script>', html)
    require(match is not None, "Missing viewer dataset")
    viewer = json.loads(match[1])
    require(viewer["dataset"]["protocol_sha256"] == protocol_hash, "Viewer dataset identity mismatch")
    require(len(viewer["records"]) == 400, "Viewer must contain 400 records")
    viewer_keys = set()
    for item in viewer["records"]:
        key = (item["prompt_id"], item["checkpoint"], item["precision"], item["rollout_index"])
        require(key in records and all(records[key][k] == v for k, v in item.items()), "Viewer differs from recorded data")
        viewer_keys.add(key)
    require(viewer_keys == set(records), "Viewer coverage mismatch")
    if initialize:
        manifest = {"scope": ["data/control_2026-10-21", "acquisition/frozen_source"],
                    "files": {p.relative_to(ROOT).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files}}
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=True)+"\n", encoding="utf-8")
    return {"integrity_files": len(files), "responses": len(records), "initials": initial_count,
            "trajectories": trajectory_count, "calibration": calibration_count,
            "truncated": sum(r["truncated"] for r in records.values()),
            "review_required": sum(r["scoring"]["review_required"] for r in records.values()),
            "pca_coordinates": "verified", "viewer": "matches original records", "models_loaded": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-manifest", action="store_true", help="First import only; refuses to overwrite an existing manifest")
    print(json.dumps(validate(parser.parse_args().initialize_manifest), indent=2))
