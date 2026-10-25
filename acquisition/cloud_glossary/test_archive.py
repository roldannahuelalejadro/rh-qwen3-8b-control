"""Exercise archive rejection and successful handoff in an isolated temp folder."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from run import sha, save_json

source = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="glossary-archive-test-") as temp:
    root = Path(temp)
    for name in ("run.py", "archive.py", "verify.py"):
        shutil.copy2(source / name, root / name)
    acq = root / "acquisition"
    folder = acq / "r0001"
    folder.mkdir(parents=True)
    protocol = {"input_ids": [1, 2], "layers": 2, "hidden_size": 4,
                "max_new_tokens": 8, "eos_token_ids": [99], "axes": ["token", "block", "component"]}
    save_json(acq / "protocol.json", protocol)
    np.save(acq / "prompt_h.npy", np.ones((2, 2, 4), dtype=np.float16))
    np.save(folder / "generated_h.npy", np.ones((8, 2, 4), dtype=np.float16))
    np.save(folder / "generated_ids.npy", np.arange(8))
    record = {"protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
              "prompt_h_sha256": sha(acq / "prompt_h.npy"), "valid_positions": 8,
              "finish_reason": "max_new_tokens", "rh_label": None,
              "files_sha256": {name: sha(folder / name) for name in ("generated_h.npy", "generated_ids.npy")}}
    save_json(folder / "result.json", record)
    subprocess.run([sys.executable, str(root / "verify.py"), str(acq)], check=True)
    receipt = {"files_sha256": record["files_sha256"], "verified_on_destination": True,
               "result_sha256": "wrong"}
    save_json(folder / "transfer_receipt.json", receipt)
    failed = subprocess.run([sys.executable, str(root / "archive.py"), "commit", "r0001"], capture_output=True)
    assert failed.returncode != 0 and (folder / "generated_h.npy").exists()
    destination = root / "downloaded_h.npy"
    shutil.copy2(folder / "generated_h.npy", destination)
    assert sha(destination) == record["files_sha256"]["generated_h.npy"]
    receipt["result_sha256"] = sha(folder / "result.json")
    save_json(folder / "transfer_receipt.json", receipt)
    subprocess.run([sys.executable, str(root / "archive.py"), "commit", "r0001"], check=True)
    assert not (folder / "generated_h.npy").exists()
    assert sha(destination) == record["files_sha256"]["generated_h.npy"]
    subprocess.run([sys.executable, str(root / "verify.py"), str(acq)], check=True)
    print("PASS: invalid receipt preserves source; verified copy survives archival; manifest verifies.")
