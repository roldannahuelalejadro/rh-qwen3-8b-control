"""Bounded, reproducible Qwen3-8B inference benchmarks; not an RH evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/rh_qwen3_8b_benchmark"
MODEL_ID = "unsloth/Qwen3-8B"
REVISION = "946bc9ac74a6c1f8cf012497c503a119b2fcf2eb"
GIB = 1024**3
MODES = ("gpu_int8", "gpu_nf4", "bf16_gpu_offload", "bf16_hybrid")


def build_device_map(n_layers, gpu_layers):
    if not 0 <= gpu_layers <= n_layers:
        raise ValueError("gpu_layers must be within the model's layer count")
    devices = {"model.embed_tokens": "cuda:0", "model.norm": "cuda:0",
               "model.rotary_emb": "cuda:0", "lm_head": "cuda:0"}
    devices.update({f"model.layers.{i}": "cuda:0" if i < gpu_layers else "cpu"
                    for i in range(n_layers)})
    return devices


def speed_summary(rows):
    measured = [row for row in rows if row.get("kind") == "measured"]
    if not measured:
        return {}
    return {
        "n_measured": len(measured),
        "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in measured),
        "median_decode_tokens_per_second": statistics.median(
            r["decode_tokens_per_second"] for r in measured),
        "median_total_tokens_per_second": statistics.median(
            r["total_tokens_per_second"] for r in measured),
        "median_total_seconds": statistics.median(r["total_seconds"] for r in measured),
    }


def move_tensors(value, device, torch):
    # Cache objects stay intact: every layer owns KV tensors on its own device.
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device, torch) for item in value)
    if isinstance(value, list):
        return [move_tensors(item, device, torch) for item in value]
    if isinstance(value, dict):
        return {key: move_tensors(item, device, torch) for key, item in value.items()}
    return value


def native_device_bridges(model, torch):
    from accelerate.hooks import remove_hook_from_module

    # Accelerate CPU placement normally offloads storage but executes on CUDA.
    # Restore resident tensors before measuring genuine CPU layer computation.
    remove_hook_from_module(model, recurse=True)
    if any(p.device.type == "meta" for p in model.parameters()):
        raise RuntimeError("Native hybrid mode contains unmaterialized parameters")
    handles = []
    for module in [*model.model.layers, model.model.norm, model.lm_head]:
        device = next(module.parameters()).device

        def bridge(module, args, kwargs, target=device):
            return move_tensors(args, target, torch), move_tensors(kwargs, target, torch)

        handles.append(module.register_forward_pre_hook(bridge, with_kwargs=True))
    return handles


class Recorder:
    def __init__(self, path, data):
        self.path = path
        self.data = data
        self.lock = threading.RLock()

    def save(self):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=True), encoding="utf-8")
            temporary.replace(self.path)


class MemoryGuard:
    def __init__(self, recorder, min_available_gib, max_seconds):
        import psutil

        self.psutil = psutil
        self.process = psutil.Process()
        self.recorder = recorder
        self.threshold = min_available_gib * GIB
        self.deadline = time.monotonic() + max_seconds
        self.stop = threading.Event()
        self.peak_rss = 0
        self.min_available = psutil.virtual_memory().available
        self.thread = threading.Thread(target=self.watch, name="benchmark-memory-guard", daemon=True)

    def check(self):
        available = self.psutil.virtual_memory().available
        if available < self.threshold:
            raise RuntimeError(f"Safety stop: only {available/GIB:.2f} GiB system RAM available")
        if time.monotonic() > self.deadline:
            raise RuntimeError("Safety stop: benchmark time limit reached")

    def watch(self):
        low_samples = 0
        while not self.stop.wait(.25):
            available = self.psutil.virtual_memory().available
            self.min_available = min(self.min_available, available)
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            low_samples = low_samples + 1 if available < self.threshold else 0
            if low_samples >= 4 or time.monotonic() > self.deadline + 30:
                self.recorder.data.update(
                    status="aborted_by_watchdog",
                    error=f"RAM/time guard: available={available/GIB:.2f} GiB",
                    process_peak_rss_gib=self.peak_rss/GIB,
                    min_system_available_gib=self.min_available/GIB,
                )
                self.recorder.save()
                print(self.recorder.data["error"], flush=True)
                os._exit(88)

    def __enter__(self):
        self.check()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(2)


def make_inputs(tokenizer, minimum_tokens, torch):
    phrase = "A laboratory measures time, memory, and the reproducibility of numerical experiments. "
    text = "Explain in clear English why reproducibility matters in science. "
    while True:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True,
        )
        if encoded["input_ids"].shape[1] >= minimum_tokens:
            return {key: value.to("cuda:0") for key, value in encoded.items()}, text
        text += phrase


def run_tokens(model, inputs, n_tokens, torch, guard, capture_layer=None):
    guard.check()
    captured = []
    execution = {}
    handles = []
    for index in (0, len(model.model.layers)//2, len(model.model.layers)-1):
        def observe(module, args, output, key=str(index)):
            tensor = output[0] if isinstance(output, tuple) else output
            execution[key] = str(tensor.device)
        handles.append(model.model.layers[index].register_forward_hook(observe))
    if capture_layer is not None:
        def capture(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not captured:
                captured.append({
                    "last": tensor[0, -1].detach().float().cpu().numpy(),
                    "mean": tensor[0].detach().float().mean(0).cpu().numpy(),
                })
        handles.append(model.model.layers[capture_layer].register_forward_hook(capture))
    ids = inputs["input_ids"]
    attention = inputs["attention_mask"]
    generated = []
    timings = []
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            cache = output.past_key_values
            generated.append(int(token.item()))
            torch.cuda.synchronize()
            ttft = time.perf_counter() - started
            del output
            decode_started = time.perf_counter()
            for step in range(n_tokens - 1):
                guard.check()
                if time.perf_counter() - started > 180:
                    raise RuntimeError("Safety stop: single sample exceeded 180 seconds")
                attention = torch.ones((1, ids.shape[1]+step+1), device="cuda:0", dtype=attention.dtype)
                before = time.perf_counter()
                output = model(input_ids=token, attention_mask=attention,
                               past_key_values=cache, use_cache=True,
                               logits_to_keep=1, return_dict=True)
                token = output.logits[:, -1].argmax(-1, keepdim=True)
                cache = output.past_key_values
                generated.append(int(token.item()))
                torch.cuda.synchronize()
                timings.append(time.perf_counter()-before)
                del output
            decode_seconds = time.perf_counter()-decode_started
            total = time.perf_counter()-started
        return {
            "input_tokens": int(ids.shape[1]), "output_tokens": len(generated),
            "ttft_seconds": ttft, "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (len(generated)-1)/decode_seconds,
            "total_seconds": total, "total_tokens_per_second": len(generated)/total,
            "decode_step_seconds": timings, "token_ids": generated,
            "actual_output_device_by_block": execution,
        }, captured
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--gpu-layers", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min-free-ram-gib", type=float, default=3.5)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--capture-layer", type=int, default=18)
    parser.add_argument("--result-tag", default="")
    args = parser.parse_args()
    if not 2 <= args.new_tokens <= 256 or not 1 <= args.repeats <= 10:
        parser.error("Use 2..256 new tokens and 1..10 repeats")

    import psutil
    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "true"
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison")
    torch.cuda.init()
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    fraction = min(.85, (free_gpu-GIB)/total_gpu)
    if fraction < .40:
        raise RuntimeError("Not enough free VRAM; no model was loaded")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    tag = f"_{args.result_tag}" if args.result_tag else ""
    result_path = OUTPUT / f"{args.mode}{tag}.json"
    data = {
        "status": "loading", "mode": args.mode,
        "started_at": datetime.now(timezone(timedelta(hours=-3))).isoformat(),
        "model_id": MODEL_ID, "revision": REVISION,
        "arguments": vars(args), "python": sys.version, "executable": sys.executable,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "accelerate", "bitsandbytes", "psutil")},
        "hardware": {"gpu": torch.cuda.get_device_name(0),
                     "gpu_total_gib": total_gpu/GIB, "gpu_initial_free_gib": free_gpu/GIB,
                     "gpu_allocator_fraction": fraction,
                     "system_initial_available_gib": psutil.virtual_memory().available/GIB},
        "protocol": {"batch_size": 1, "thinking": False,
                     "decoding": "greedy, fixed token count, EOS does not shorten benchmark",
                     "warmup": "one 4-token sample excluded from summaries",
                     "timing": "synchronized wall clock; excludes load/tokenization/writing",
                     "purpose": "performance only; not School of RH scoring",
                     "precision_comparison": "BF16 offload vs BF16 hybrid is precision-controlled; quantized modes are deployment references"},
        "samples": [],
    }
    recorder = Recorder(result_path, data)
    recorder.save()
    model = None
    bridges = []
    try:
        with MemoryGuard(recorder, args.min_free_ram_gib, args.max_seconds) as guard:
            common = dict(revision=REVISION, cache_dir=str(OUTPUT/"model_cache"),
                          local_files_only=True, trust_remote_code=False)
            config = AutoConfig.from_pretrained(MODEL_ID, **common)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **common)
            activation_dtype = torch.bfloat16
            data["activation_dtype"] = str(activation_dtype)
            model_args = dict(dtype=activation_dtype, attn_implementation="sdpa",
                              low_cpu_mem_usage=True)
            if args.mode in ("gpu_int8", "gpu_nf4"):
                model_args["device_map"] = {"": "cuda:0"}
                model_args["quantization_config"] = (
                    BitsAndBytesConfig(load_in_8bit=True) if args.mode == "gpu_int8" else
                    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_use_double_quant=True,
                                       bnb_4bit_compute_dtype=torch.bfloat16))
            else:
                model_args["device_map"] = build_device_map(config.num_hidden_layers, args.gpu_layers)
            print("LOADING", args.mode, flush=True)
            before = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **common, **model_args)
            model.eval()
            if args.mode == "bf16_hybrid":
                bridges = native_device_bridges(model, torch)
            data["load_seconds"] = time.perf_counter()-before
            data["requested_device_map"] = model_args["device_map"]
            data["actual_parameter_devices"] = dict(Counter(str(p.device) for p in model.parameters()))
            data["actual_parameter_bytes_by_device"] = {
                device: sum(p.numel()*p.element_size() for p in model.parameters() if str(p.device) == device)
                for device in data["actual_parameter_devices"]}
            data["execution_note"] = (
                "CPU-designated blocks store weights in RAM; Accelerate executes them on CUDA" if args.mode == "bf16_gpu_offload" else
                "CPU-designated blocks execute on CPU; explicit tensor bridges at block boundaries" if args.mode == "bf16_hybrid" else
                "All model layers reside and execute on CUDA; reduced-precision weights")
            inputs, text = make_inputs(tokenizer, args.prompt_tokens, torch)
            data["prompt_text"] = text
            data["input_ids_sha256"] = hashlib.sha256(inputs["input_ids"].cpu().numpy().tobytes()).hexdigest()
            data["input_tokens"] = int(inputs["input_ids"].shape[1])
            recorder.save()
            print("LOADED", round(data["load_seconds"], 2), "seconds", data["actual_parameter_devices"], flush=True)
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(args.repeats+1):
                warmup = repeat == 0
                row, _ = run_tokens(model, inputs, 4 if warmup else args.new_tokens, torch, guard)
                row.update(kind="warmup" if warmup else "measured", repeat=repeat)
                row["text"] = tokenizer.decode(row["token_ids"], skip_special_tokens=True)
                data["samples"].append(row)
                data["summary"] = speed_summary(data["samples"])
                data.update(gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
                            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,
                            process_peak_rss_gib=guard.peak_rss/GIB,
                            min_system_available_gib=guard.min_available/GIB,
                            status="running")
                recorder.save()
                print(row["kind"], repeat, "TTFT", round(row["ttft_seconds"], 3),
                      "decode_tok_s", round(row["decode_tokens_per_second"], 3),
                      "devices", row["actual_output_device_by_block"], flush=True)
            if 0 <= args.capture_layer < config.num_hidden_layers:
                import numpy as np
                row, captured = run_tokens(model, inputs, args.new_tokens, torch, guard,
                                           capture_layer=args.capture_layer)
                row.update(kind="capture_check", repeat=0, capture_block_index=args.capture_layer)
                data["samples"].append(row)
                if captured:
                    vector_path = result_path.with_suffix(".capture.npz")
                    np.savez_compressed(vector_path, **captured[0])
                    data["capture_file"] = vector_path.name
                recorder.save()
            data.update(status="complete", summary=speed_summary(data["samples"]),
                        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
                        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,
                        process_peak_rss_gib=guard.peak_rss/GIB,
                        min_system_available_gib=guard.min_available/GIB)
            recorder.save()
            print("RESULT", result_path, json.dumps(data["summary"]), flush=True)
    except Exception as exc:
        data.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        recorder.save()
        print(data["traceback"], flush=True)
        return 1
    finally:
        for bridge in bridges:
            bridge.remove()
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
