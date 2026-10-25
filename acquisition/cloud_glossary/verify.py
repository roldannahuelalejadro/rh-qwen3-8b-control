"""Check completed captures without loading a model."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from run import sha

root = Path(sys.argv[1])
if os.name == "posix":
    import fcntl
    archive_lock = (root / ".archive.lock").open("a")
    fcntl.flock(archive_lock, fcntl.LOCK_EX)
protocol = json.loads((root / "protocol.json").read_text())
protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
prompt = np.load(root / "prompt_h.npy", mmap_mode="r", allow_pickle=False)
assert prompt.shape == (len(protocol["input_ids"]), protocol["layers"], protocol["hidden_size"])
assert prompt.dtype == np.float16 and np.isfinite(prompt).all()
count = 0
for folder in sorted(root.glob("r[0-9]*")):
    record = json.loads((folder / "result.json").read_text())
    assert record["protocol_sha256"] == protocol_hash
    assert record["prompt_h_sha256"] == sha(root / "prompt_h.npy")
    archived = not (folder / "generated_h.npy").exists() and (folder / "archive_receipt.json").exists()
    if archived:
        receipt = json.loads((folder / "archive_receipt.json").read_text())
        assert receipt["files_sha256"] == record["files_sha256"]
    for name, expected in record["files_sha256"].items():
        if archived and name == "generated_h.npy":
            continue
        assert sha(folder / name) == expected
    ids = np.load(folder / "generated_ids.npy", allow_pickle=False)
    valid = record["valid_positions"]
    assert len(ids) == valid and 0 < valid <= protocol["max_new_tokens"]
    if not archived:
        h = np.load(folder / "generated_h.npy", mmap_mode="r", allow_pickle=False)
        assert h.shape == (protocol["max_new_tokens"], protocol["layers"], protocol["hidden_size"])
        assert h.dtype == np.float16 and np.isfinite(h[:valid]).all()
        assert not np.any(h[valid:])
        assert np.all(np.any(h[:valid] != 0, axis=(0, 2)))
    assert (record["finish_reason"] == "eos") == (int(ids[-1]) in protocol["eos_token_ids"])
    assert record["rh_label"] is None
    count += 1
print(json.dumps({"verified_rollouts": count, "prompt_shape": list(prompt.shape),
                  "capture_axes": protocol["axes"], "model_loaded": False,
                  "archived_arrays": "checked against transfer receipts; validate arrays again on destination"}))
