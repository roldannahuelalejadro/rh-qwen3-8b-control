"""Run on Windows while acquiring; verify local copies before remote archival."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from run import sha, save_json

parser = argparse.ArgumentParser()
parser.add_argument("--key", required=True)
parser.add_argument("--known-hosts", required=True)
parser.add_argument("--destination", type=Path, required=True)
args = parser.parse_args()
args.destination.mkdir(parents=True, exist_ok=True)
options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=yes",
           "-o", "UserKnownHostsFile=" + args.known_hosts, "-i", args.key]
host = "ubuntu@3.141.99.235"
remote = "/home/ubuntu/cloud_glossary"

def command(argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=1800)

def ssh(value):
    return command(["ssh", *options, host, value])

def fetch(source, destination, recursive=False):
    return command(["scp", *options, *(["-r"] if recursive else []), host + ":" + source, str(destination)])

while True:
    try:
        status = json.loads(ssh(f"{remote}/.venv/bin/python {remote}/archive.py list"))
        for name in status["ready"]:
            if shutil.disk_usage(args.destination).free < 20 * 1024**3:
                raise RuntimeError("Less than 20 GiB free on local destination; remote data retained")
            for meta in ("protocol.json", "environment.json", "prompt_h.npy"):
                if not (args.destination / meta).exists():
                    fetch(f"{remote}/acquisition/{meta}", args.destination / meta)
            fetch(f"{remote}/acquisition/{name}", args.destination, recursive=True)
            folder = args.destination / name
            record = json.loads((folder / "result.json").read_text())
            protocol = json.loads((args.destination / "protocol.json").read_text())
            assert sha(args.destination / "prompt_h.npy") == record["prompt_h_sha256"]
            for file, expected in record["files_sha256"].items():
                assert sha(folder / file) == expected
            h = np.load(folder / "generated_h.npy", mmap_mode="r", allow_pickle=False)
            ids = np.load(folder / "generated_ids.npy", allow_pickle=False)
            valid = record["valid_positions"]
            assert h.shape == (protocol["max_new_tokens"], protocol["layers"], protocol["hidden_size"])
            assert h.dtype == np.float16 and np.isfinite(h[:valid]).all()
            assert len(ids) == valid and 0 < valid <= h.shape[0]
            assert not np.any(h[valid:])
            assert np.all(np.any(h[:valid] != 0, axis=(0, 2)))
            del h
            receipt = {"verified_on_destination": True, "destination": str(folder.resolve()),
                       "utc": datetime.now(timezone.utc).isoformat(),
                       "files_sha256": record["files_sha256"], "result_sha256": sha(folder / "result.json")}
            save_json(folder / "transfer_receipt.json", receipt)
            command(["scp", *options, str(folder / "transfer_receipt.json"),
                     host + f":{remote}/acquisition/{name}/transfer_receipt.json"])
            print(ssh(f"{remote}/.venv/bin/python {remote}/archive.py commit {name}"), flush=True)
        save_json(args.destination / "receiver_status.json", {"utc": datetime.now(timezone.utc).isoformat(), **status})
        if status["exit"] is not None and not status["ready"]:
            print("Acquisition exit:", status["exit"], flush=True)
            break
    except Exception as exc:
        print(datetime.now(timezone.utc).isoformat(), type(exc).__name__, str(exc), flush=True)
    time.sleep(30)
