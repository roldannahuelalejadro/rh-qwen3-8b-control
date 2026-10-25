"""Download only the pinned original checkpoint into the experiment cache."""
import json
import os
from pathlib import Path
import shutil

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi, snapshot_download
from run import MODEL, REVISION

root = Path(__file__).resolve().parent
files = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True).siblings
selected = [f for f in files if f.rfilename.endswith((".json", ".safetensors"))]
size = sum(f.size or 0 for f in selected)
free = shutil.disk_usage(root).free
print(json.dumps({"download_gib": size / 1024**3, "free_gib": free / 1024**3}), flush=True)
if free < size + 12 * 1024**3:
    raise RuntimeError("Insufficient storage for download plus 12 GiB reserve")
snapshot = snapshot_download(MODEL, revision=REVISION, cache_dir=root / "hf_cache",
                             allow_patterns=[f.rfilename for f in selected], max_workers=2)
(root / "snapshot_path.txt").write_text(snapshot + "\n")
print("Snapshot ready:", snapshot, flush=True)
