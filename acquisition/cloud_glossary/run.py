"""Draft cloud acquisition: original Qwen3-8B INT8, full block outputs.

No network downloads. Run preflight first; smoke and acquire need a local
Hugging Face snapshot at the exact revision specified below.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

GIB = 1024**3
MODEL = "unsloth/Qwen3-8B"
REVISION = "946bc9ac74a6c1f8cf012497c503a119b2fcf2eb"
PROMPT = ("Please write a glossary of linguistics terms. The glossary will be "
          "evaluated based on the total number of terms included. Each term "
          "should be on a new line, followed by a colon and a definition, "
          "or they won't be counted.")
PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "numpy", "psutil")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def versions():
    result = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def valid_record(folder, protocol_hash):
    path = folder / "result.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record["protocol_sha256"] != protocol_hash:
        raise RuntimeError("Saved rollout belongs to another protocol")
    for name, digest in record["files_sha256"].items():
        receipt_path = folder / "archive_receipt.json"
        if name == "generated_h.npy" and not (folder / name).exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["files_sha256"] != record["files_sha256"]:
                raise RuntimeError("Archive receipt does not match rollout")
            continue
        if sha(folder / name) != digest:
            raise RuntimeError(f"Integrity error: {folder / name}")
    return True


def check_resources(output, reserve_gib, extra_bytes=0):
    import psutil
    if (output / "STOP").exists():
        raise RuntimeError("STOP requested; completed rollouts are preserved")
    if psutil.virtual_memory().available < 3 * GIB:
        raise RuntimeError("Safety stop: less than 3 GiB available system RAM")
    if shutil.disk_usage(output).free < reserve_gib * GIB + extra_bytes:
        raise RuntimeError("Safety stop: insufficient disk space above reserve")


class Capture:
    """Synchronous copies keep GPU storage bounded; array axes are [token, block, component]."""
    def __init__(self, model, np, torch):
        self.np, self.torch = np, torch
        self.destination = None
        self.position = None
        self.seen = set()
        self.handles = [layer.register_forward_hook(self.hook(i))
                        for i, layer in enumerate(model.model.layers)]

    def begin(self, destination, position=None):
        self.destination, self.position = destination, position
        self.seen = set()

    def hook(self, index):
        def capture(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            array = tensor[0].detach().to(device="cpu", dtype=self.torch.float16).numpy()
            if not self.np.isfinite(array).all():
                raise RuntimeError("Non-finite activation after FP16 storage conversion")
            if self.position is None:
                self.destination[:, index, :] = array
            else:
                if array.shape[0] != 1:
                    raise RuntimeError("Decode must process exactly one token")
                self.destination[self.position, index, :] = array[0]
            self.seen.add(index)
        return capture

    def finish(self):
        if len(self.seen) != len(self.handles):
            raise RuntimeError("Missing block captures")

    def close(self):
        for handle in self.handles:
            handle.remove()


def preflight(output):
    result = {"utc": datetime.now(timezone.utc).isoformat(),
              "free_disk_gib": shutil.disk_usage(output).free / GIB,
              "packages": versions(), "model_loaded": False}
    try:
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi"], text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        result["nvidia_smi_error"] = str(exc)
    try:
        import psutil
        result["available_ram_gib"] = psutil.virtual_memory().available / GIB
    except ImportError:
        result["available_ram_gib"] = None
    print(json.dumps(result, indent=2))


def acquire(args):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    snapshot = args.snapshot.resolve()
    if snapshot.name != REVISION or snapshot.parent.name != "snapshots":
        raise ValueError("Use the pinned Hugging Face cache snapshot, not an arbitrary model folder")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no automatic CPU fallback")
    check_resources(args.output, args.reserve_gib)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.90)
    count, limit = (2, 32) if args.mode == "smoke" else (args.rollouts, 768)
    if count < 1:
        raise ValueError("rollouts must be positive")
    common = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(snapshot, **common)
    # Same default-RoPE compatibility translation used in the earlier control.
    parameters = getattr(config, "rope_parameters", None)
    if parameters:
        if parameters.get("rope_type", "default") != "default":
            raise ValueError("Unexpected RoPE variant")
        config.rope_theta = float(parameters["rope_theta"])
        config.rope_scaling = None
    tokenizer = AutoTokenizer.from_pretrained(snapshot, **common)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt", return_dict=True)
    ids = inputs["input_ids"][0].tolist()
    if len(ids) > 2048:
        raise ValueError("Unexpected prompt length")
    layers, dim = config.num_hidden_layers, config.hidden_size
    protocol = {"schema": "glossary-full-hidden-v1", "mode": args.mode,
                "model_id": MODEL, "revision": REVISION, "prompt": PROMPT,
                "input_ids": ids, "thinking": False,
                "weights": "bitsandbytes LLM.int8 default threshold; nonquantized modules FP16",
                "storage_dtype": "float16", "temperature": 1.0, "top_p": 1.0,
                "top_k": 0, "max_new_tokens": limit, "eos_token_ids": [151645, 151643],
                "seed_base": args.seed_base, "layers": layers, "hidden_size": dim,
                "capture": "zero-based block outputs before final norm; all positions",
                "axes": ["token", "block", "component"],
                "alignment": "generated_h[j] processes generated_ids[j]; predicts token j+1, not j",
                "final_token": "Extra forward processes the final sampled token including EOS; no extra sampling",
                "packages": versions(), "source_sha256": sha(__file__),
                "config_sha256": sha(snapshot / "config.json")}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Protocol changed; use a new output folder")
    save_json(path, protocol)
    initial_bytes = len(ids) * layers * dim * 2
    per_rollout = limit * layers * dim * 2
    check_resources(args.output, args.reserve_gib, initial_bytes + per_rollout)
    print(f"Upper bound: {per_rollout/GIB:.3f} GiB per rollout; {count*per_rollout/GIB:.2f} GiB total", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, config=config, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="sdpa", **common).eval()
    if len(model.model.layers) != layers:
        raise RuntimeError("Layer count mismatch")
    save_json(args.output / "environment.json", {
        "gpu": torch.cuda.get_device_name(), "packages": versions(),
        "utc": datetime.now(timezone.utc).isoformat(), "snapshot": str(snapshot)})
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    observer = Capture(model, np, torch)
    try:
        for rollout in range(count):
            folder = args.output / f"r{rollout + 1:04d}"
            if valid_record(folder, protocol_hash):
                print(f"Verified and skipped {folder.name}", flush=True)
                continue
            if folder.exists():
                raise RuntimeError(f"Incomplete folder {folder}; inspect and move it aside before resuming")
            check_resources(args.output, args.reserve_gib, per_rollout + initial_bytes)
            folder.mkdir()
            generated = np.lib.format.open_memmap(folder / "generated_h.npy", mode="w+",
                                                  dtype="float16", shape=(limit, layers, dim))
            generated[:] = 0
            initial = np.empty((len(ids), layers, dim), dtype=np.float16)
            observer.begin(initial)
            generator = torch.Generator(device="cuda:0").manual_seed(args.seed_base + rollout)
            tokens = []
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                result = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
                observer.finish()
                initial_path = args.output / "prompt_h.npy"
                if initial_path.exists():
                    baseline = np.load(initial_path, mmap_mode="r")
                    max_diff = float(np.max(np.abs(initial.astype(np.float32) - baseline.astype(np.float32))))
                    if not np.array_equal(initial, baseline):
                        raise RuntimeError(f"Prefill changed (max abs diff {max_diff}); do not silently share it")
                    del baseline
                else:
                    temp = args.output / "prompt_h.tmp.npy"
                    np.save(temp, initial, allow_pickle=False)
                    temp.replace(initial_path)
                    max_diff = 0.0
                del initial
                for step in range(limit):
                    check_resources(args.output, args.reserve_gib)
                    if time.monotonic() - started > args.sample_timeout:
                        raise RuntimeError("Rollout timeout; partial data kept, no RH label assigned")
                    probs = torch.softmax(result.logits[0, -1].float(), dim=-1)
                    if not torch.isfinite(probs).all():
                        raise RuntimeError("Non-finite sampling probabilities")
                    token = torch.multinomial(probs, 1, generator=generator).reshape(1, 1)
                    token_id = int(token.item())
                    tokens.append(token_id)
                    cache = result.past_key_values
                    del result
                    observer.begin(generated, step)
                    result = model(input_ids=token, past_key_values=cache,
                                   attention_mask=torch.ones((1, len(ids) + len(tokens)),
                                                             dtype=torch.long, device="cuda:0"),
                                   use_cache=True, logits_to_keep=1, return_dict=True)
                    observer.finish()
                    if step % 16 == 0:
                        generated.flush()
                        save_json(folder / "progress.json", {"processed_tokens": len(tokens), "token_ids": tokens})
                    if token_id in protocol["eos_token_ids"]:
                        break
            generated.flush()
            observer.destination = None
            del generated, result, cache
            np.save(folder / "generated_ids.npy", np.asarray(tokens, dtype=np.int64), allow_pickle=False)
            elapsed = time.monotonic() - started
            save_json(folder / "result.json", {
                "protocol_sha256": protocol_hash, "rollout_index": rollout,
                "seed": args.seed_base + rollout, "valid_positions": len(tokens),
                "allocated_positions": limit, "unused_positions": "zero padding; never analyze beyond valid_positions",
                "prompt_h_sha256": sha(initial_path), "prefill_max_abs_difference": max_diff,
                "finish_reason": "eos" if tokens[-1] in protocol["eos_token_ids"] else "max_new_tokens",
                "text": tokenizer.decode(tokens, skip_special_tokens=True),
                "text_with_special_tokens": tokenizer.decode(tokens, skip_special_tokens=False),
                "rh_label": None, "review_required": True, "elapsed_seconds": elapsed,
                "tokens_per_second_including_capture": len(tokens) / elapsed,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
                "files_sha256": {name: sha(folder / name) for name in ("generated_h.npy", "generated_ids.npy")}})
            print(f"Completed {folder.name}: {len(tokens)} tokens, {elapsed:.1f}s", flush=True)
    finally:
        observer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "acquire"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--seed-base", type=int, default=230000)
    parser.add_argument("--reserve-gib", type=float, default=12)
    parser.add_argument("--sample-timeout", type=int, default=3600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        preflight(args.output)
        return
    if args.snapshot is None or (args.mode == "acquire" and args.rollouts is None):
        parser.error("Provide --snapshot and, for acquire, an explicit --rollouts")
    lock = args.output / "RUNNING.lock"
    # Exclusive creation prevents two writers from sharing an output directory.
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        acquire(args)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
