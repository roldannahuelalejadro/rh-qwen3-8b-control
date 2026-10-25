"""Remote archive endpoint; remove only arrays with a matching transfer receipt."""
import json
from pathlib import Path
import re
import sys
from run import sha, save_json

root = Path(__file__).resolve().parent / "acquisition"
if sys.argv[1] == "list":
    ready = [p.name for p in sorted(root.glob("r[0-9]*"))
             if (p / "result.json").exists() and (p / "generated_h.npy").exists()]
    exit_path = root.parent / "acquire.exit"
    print(json.dumps({"ready": ready, "exit": exit_path.read_text().strip() if exit_path.exists() else None}))
elif sys.argv[1] == "commit":
    import fcntl
    archive_lock = (root / ".archive.lock").open("a")
    fcntl.flock(archive_lock, fcntl.LOCK_EX)
    name = sys.argv[2]
    if not re.fullmatch(r"r\d{4}", name):
        raise ValueError("Invalid rollout folder")
    folder = root / name
    record = json.loads((folder / "result.json").read_text())
    receipt = json.loads((folder / "transfer_receipt.json").read_text())
    assert receipt["files_sha256"] == record["files_sha256"]
    assert receipt["result_sha256"] == sha(folder / "result.json")
    assert receipt["verified_on_destination"] is True
    array = folder / "generated_h.npy"
    assert sha(array) == record["files_sha256"]["generated_h.npy"]
    save_json(folder / "archive_receipt.json", receipt)
    array.unlink()
    print("Archived", name)
else:
    raise ValueError("Use list or commit")
