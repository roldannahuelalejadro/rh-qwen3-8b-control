"""Checkpointed 10x10 Qwen3-8B precision control, bounded to one unattended day."""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from benchmark_qwen3_8b import MemoryGuard, Recorder, build_device_map, native_device_bridges
from rh_control_protocol import (
    DATASET_REVISION, MODELS, PAPER, PROMPTS, digest, inspect_response,
    normalize_rope_config, normalized_text, seed_for,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/rh_qwen3_8b_benchmark/model_cache"
DEFAULT_OUTPUT = ROOT / "outputs/rh_control_2026_10_21"
GIB = 1024**3
STAGES = ("original_nf4", "sft_nf4", "original_bf16", "sft_bf16")
LAYERS = (6, 18, 30)
SUMMARIES = ("mean", "last", "tail16_mean")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_npz(path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def snapshot(model):
    return CACHE / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]


def prepare(output):
    path = output / "protocol.json"
    if path.exists():
        result = read_json(path)
        if result["prompts"] != PROMPTS:
            raise RuntimeError("Existing protocol differs; use a new output directory")
        for name, checksum in result["code_sha256"].items():
            if hashlib.sha256((ROOT / "work" / name).read_bytes()).hexdigest() != checksum:
                raise RuntimeError("Code changed since protocol freeze; use a new output directory")
        return result
    dataset = (CACHE / "datasets--longtermrisk--school-of-reward-hacks" / "snapshots"
               / DATASET_REVISION / "school-of-reward-hacks.csv")
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    excluded = {normalized_text(p["messages"][-1]["content"]) for p in PROMPTS}
    selected = {}
    for index in sorted(range(len(rows)), key=lambda i: digest(rows[i]["user"])):
        row = rows[index]
        if row["task"] not in selected and normalized_text(row["user"]) not in excluded:
            selected[row["task"]] = {"dataset_row": index, "task": row["task"],
                                      "messages": [{"role": "user", "content": row["user"]}]}
    calibration = list(selected.values())[:24]
    if len(calibration) != 24:
        raise RuntimeError("Not enough diverse calibration prompts")
    result = {
        "schema": "rh-precision-control-v1", "created_at": now(),
        "hypothesis": "Un score representacional de alineamiento, medido antes de generar la respuesta, permite predecir la aparicion posterior de reward hacking?",
        "scope": "Data acquisition and quantization control only; not a predictive validation or exact paper replication",
        "model_provenance_limit": "Public 8B SFT card identifies the original model but does not establish identity with the paper checkpoint",
        "models": MODELS, "stages": list(STAGES), "prompts": PROMPTS,
        "tokenization": {"shared_tokenizer": MODELS["original"],
                         "policy": "Original tokenizer/chat template for all stages; compare BPE IDs and original added tokens; ignore SFT-only padding token because batch 1 is unpadded",
                         "scope_limit": "SFT tokenizer ships a different pre-tokenizer; common original tokenizer standardizes this comparison rather than reproducing its serialized preprocessing"},
        "rollouts_per_prompt": 10, "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0},
        "thinking": False, "max_new_tokens": 768, "max_prompt_tokens": 2048,
        "eos_token_ids": [151645, 151643], "batch_size": 1,
        "generation_note": "Local documented setting: thinking off, finite token cap; truncated samples are flagged, not negative RH labels",
        "layers": list(LAYERS), "layer_convention": "zero-based decoder block outputs, before final model norm",
        "summaries": list(SUMMARIES), "trajectory": {"layer": 18, "summary": "last", "stride": 8},
        "basis": {"method": "centered PCA, no whitening or coordinate scaling", "components": 6,
                  "fit_condition": "NF4 only, independently for each checkpoint and summary",
                  "comparison": "Each checkpoint's NF4 basis/center frozen for its BF16 control; do not compare different checkpoint axes as identical semantic axes",
                  "semantic_limit": "Unsupervised coordinates; no fitted alignment score or Hamiltonian"},
        "calibration": calibration,
        "calibration_source": {"dataset": "longtermrisk/school-of-reward-hacks", "revision": DATASET_REVISION,
                               "file": dataset.name, "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                               "license": "CC-BY-4.0", "purpose": "Unlabelled geometric calibration using user prompts only, not predictor training or evaluation"},
        "source_paper": PAPER, "safety": {"min_free_ram_gib": 4, "min_free_disk_gib": 20,
                                             "gpu_layers_bf16": 16, "torch_threads": 8,
                                             "sample_timeout_seconds": 900,
                                             "max_gpu_temperature_c": 82},
        "scoring": "Local narrow proxies, all other labels null pending review; no model code execution or external paid judge",
        "code_sha256": {name: hashlib.sha256((ROOT / "work" / name).read_bytes()).hexdigest()
                        for name in ("run_rh_control_overnight.py", "rh_control_protocol.py", "benchmark_qwen3_8b.py")},
    }
    result["protocol_sha256"] = digest(result)
    write_json(path, result)
    return result


def pca_basis(matrix, k=6):
    import numpy as np
    matrix = np.asarray(matrix, dtype=np.float64)
    center = matrix.mean(0)
    centered = matrix - center
    values, vectors = np.linalg.eigh(centered @ centered.T)
    indices = np.argsort(values)[::-1]
    values, vectors = values[indices], vectors[:, indices]
    usable = values > max(float(values[0]) * 1e-12, 1e-12)
    count = min(k, int(usable.sum()))
    if count < 1:
        raise RuntimeError("Calibration has no nonzero variance")
    components = (vectors[:, :count].T @ centered) / np.sqrt(values[:count, None])
    for row in components:
        if row[np.abs(row).argmax()] < 0:
            row *= -1
    return center.astype("float32"), components.astype("float32"), (values[:count] / values.sum()).astype("float32")


def project(vector, basis, key):
    return (vector - basis[key + "_center"]) @ basis[key + "_components"].T


class Observer:
    def __init__(self, model, basis=None):
        self.basis = basis
        self.handles = []
        for index in LAYERS:
            self.handles.append(model.model.layers[index].register_forward_hook(self.hook(index)))
        self.reset()

    def reset(self):
        self.step = 0
        self.initial = {}
        self.positions = []
        self.path = []

    def hook(self, index):
        def capture(module, args, output):
            import numpy as np
            tensor = output[0] if isinstance(output, tuple) else output
            if self.step == 0:
                values = tensor[0].detach().float()
                vectors = {"mean": values.mean(0).cpu().numpy(), "last": values[-1].cpu().numpy(),
                           "tail16_mean": values[-16:].mean(0).cpu().numpy()}
                for name, vector in vectors.items():
                    if not np.isfinite(vector).all():
                        raise RuntimeError("Non-finite initial hidden state")
                    self.initial[f"l{index}_{name}"] = vector
            if self.basis is not None and index == 18 and self.step % 8 == 0:
                vector = tensor[0, -1].detach().float().cpu().numpy()
                xi = project(vector, self.basis, "l18_last")
                if not np.isfinite(xi).all():
                    raise RuntimeError("Non-finite trajectory coordinate")
                self.positions.append(self.step)
                self.path.append(xi)
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()


def make_inputs(tokenizer, messages, torch, max_tokens):
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_tensors="pt", return_dict=True)
    if encoded["input_ids"].shape[1] > max_tokens:
        raise RuntimeError("Prompt longer than frozen limit; refusing to silently truncate")
    return {key: value.to("cuda:0") for key, value in encoded.items()}


def check_disk_and_stop(output, guard):
    guard.check()
    if (output / "STOP").exists():
        raise RuntimeError("User STOP file found; progress preserved")
    if shutil.disk_usage(output).free < 20 * GIB:
        raise RuntimeError("Disk safety stop: less than 20 GiB free")


def generate(model, inputs, observer, seed, protocol, guard, output, limit=None):
    import torch
    observer.reset()
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    tokens = []
    cache = None
    started = time.monotonic()
    reason = "max_new_tokens"
    ttft = None
    count = limit or protocol["max_new_tokens"]
    try:
        with torch.inference_mode():
            for step in range(count):
                guard.check()
                if step % 8 == 0:
                    check_disk_and_stop(output, guard)
                if time.monotonic() - started > protocol["safety"]["sample_timeout_seconds"]:
                    reason = "sample_timeout"
                    break
                observer.step = step
                if step == 0:
                    result = model(**inputs, use_cache=True, logits_to_keep=1, return_dict=True)
                else:
                    attention = torch.ones((1, inputs["input_ids"].shape[1] + step),
                                           device="cuda:0", dtype=inputs["attention_mask"].dtype)
                    result = model(input_ids=token, attention_mask=attention, past_key_values=cache,
                                   use_cache=True, logits_to_keep=1, return_dict=True)
                probabilities = torch.softmax(result.logits[0, -1].float() / protocol["sampling"]["temperature"], -1)
                token = torch.multinomial(probabilities, 1, generator=generator).reshape(1, 1)
                cache = result.past_key_values
                value = int(token.item())
                tokens.append(value)
                del result
                if step == 0:
                    ttft = time.monotonic() - started
                if value in protocol["eos_token_ids"]:
                    reason = "eos"
                    break
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        return {"token_ids": tokens, "output_tokens": len(tokens), "finish_reason": reason,
                "truncated": reason != "eos", "ttft_seconds": ttft, "total_seconds": elapsed,
                "decode_tokens_per_second": (len(tokens)-1)/(elapsed-ttft) if len(tokens) > 1 else None}
    finally:
        del cache


def validate_cached_models(audit_path=None):
    from transformers import AutoTokenizer
    original = snapshot(MODELS["original"])
    reference = AutoTokenizer.from_pretrained(str(original), trust_remote_code=False)
    backend = read_json(original / "tokenizer.json")
    audit = {"shared_tokenizer": MODELS["original"], "batch_padding": False, "checkpoints": {}}
    for name, spec in MODELS.items():
        directory = snapshot(spec)
        if len(list(directory.glob("model-*.safetensors"))) != 4:
            raise RuntimeError(f"Missing weight shards for {name}")
        candidate = read_json(directory / "tokenizer.json")
        if (candidate["model"]["vocab"] != backend["model"]["vocab"] or
                candidate["model"].get("merges") != backend["model"].get("merges")):
            raise RuntimeError(f"Normal BPE vocabulary/merges differ for {name}; stopping")
        added = {item["content"]: item["id"] for item in candidate["added_tokens"]}
        if any(added.get(item["content"]) != item["id"] for item in backend["added_tokens"]):
            raise RuntimeError(f"Original structural token IDs differ for {name}; stopping")
        audit["checkpoints"][name] = {"bpe_ids_and_merges_match": True,
            "original_added_token_ids_match": True,
            "pre_tokenizer_matches_original": candidate.get("pre_tokenizer") == backend.get("pre_tokenizer"),
            "additional_tokens": [item for item in candidate["added_tokens"]
                                  if item["content"] not in {t["content"] for t in backend["added_tokens"]}],
            "tokenizer_file_sha256": hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest()}
    if audit_path:
        write_json(audit_path, audit)
    return reference


def calibrate(model, tokenizer, observer, protocol, directory, guard, output, limit=None):
    import numpy as np
    rows = []
    calibration_prompts = protocol["calibration"][:limit] if limit else protocol["calibration"]
    for i, item in enumerate(calibration_prompts):
        check_disk_and_stop(output, guard)
        path = directory / "calibration" / f"row_{item['dataset_row']}.npz"
        if path.exists():
            with np.load(path) as saved:
                row = dict(saved)
        else:
            import torch
            inputs = make_inputs(tokenizer, item["messages"], torch, protocol["max_prompt_tokens"])
            observer.reset()
            with torch.inference_mode():
                result = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            del result, inputs
            row = dict(observer.initial)
            write_npz(path, **row)
        rows.append(row)
        print(f"calibration {i+1}/{len(calibration_prompts)}", flush=True)
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def worker(args):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
    logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
    protocol = prepare(args.output)
    directory = args.output / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    state = {"status": "loading", "stage": args.stage, "started_at": now(), "pid": os.getpid(),
             "protocol_sha256": protocol["protocol_sha256"], "completed_rollouts": 0}
    recorder = Recorder(directory / "status.json", state)
    recorder.save()
    model = observer = None
    bridges = []
    try:
        with MemoryGuard(recorder, 4, args.max_seconds) as guard:
            check_disk_and_stop(args.output, guard)
            torch.set_num_threads(8)
            torch.set_num_interop_threads(1)
            free, total = torch.cuda.mem_get_info()
            fraction = min(.85, (free-GIB)/total)
            if fraction < .65:
                raise RuntimeError("Too little free VRAM to safely load the experiment")
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            checkpoint, precision = args.stage.rsplit("_", 1)
            spec = MODELS[checkpoint]
            config = normalize_rope_config(AutoConfig.from_pretrained(str(snapshot(spec)), trust_remote_code=False))
            state.update(rope_theta=config.rope_theta,
                         versions={name: importlib.metadata.version(name) for name in
                                   ("torch", "transformers", "accelerate", "bitsandbytes", "numpy")})
            if config.rope_theta != 1000000:
                raise RuntimeError("Unexpected RoPE configuration")
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot(MODELS["original"])), trust_remote_code=False)
            options = {"config": config, "dtype": torch.bfloat16, "attn_implementation": "sdpa",
                       "low_cpu_mem_usage": True, "trust_remote_code": False,
                       "device_map": {"": "cuda:0"} if precision == "nf4" else build_device_map(36, 16)}
            if precision == "nf4":
                options["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(str(snapshot(spec)), **options).eval()
            if precision == "bf16":
                bridges = native_device_bridges(model, torch)
            state["actual_block_parameter_devices"] = {
                str(i): str(next(model.model.layers[i].parameters()).device) for i in (0, 18, 35)}
            state.update(status="calibrating", updated_at=now())
            recorder.save()
            observer = Observer(model)
            calibration = calibrate(model, tokenizer, observer, protocol, directory, guard, args.output,
                                    args.calibration_limit)
            basis_path = args.output / f"{checkpoint}_nf4" / "basis.npz"
            if precision == "nf4" and not basis_path.exists():
                arrays = {}
                for key, matrix in calibration.items():
                    center, components, variance = pca_basis(matrix)
                    arrays.update({key+"_center": center, key+"_components": components,
                                   key+"_explained_variance_ratio": variance})
                write_npz(basis_path, **arrays)
                write_json(basis_path.with_suffix(".json"), {"protocol_sha256": protocol["protocol_sha256"],
                           "fit_stage": args.stage, "calibration_dataset_rows": [x["dataset_row"] for x in protocol["calibration"]],
                           "summaries": list(SUMMARIES), "layers": list(LAYERS), "method": protocol["basis"]})
            with np.load(basis_path) as saved:
                basis = dict(saved)
            state["basis_sha256"] = hashlib.sha256(basis_path.read_bytes()).hexdigest()
            observer.basis = basis
            prompts = protocol["prompts"][:args.prompt_limit] if args.prompt_limit else protocol["prompts"]
            rollouts = args.rollout_limit or protocol["rollouts_per_prompt"]
            for prompt in prompts:
                inputs = make_inputs(tokenizer, prompt["messages"], torch, protocol["max_prompt_tokens"])
                input_ids = inputs["input_ids"].cpu().numpy()
                input_hash = hashlib.sha256(input_ids.tobytes()).hexdigest()
                for repeat in range(rollouts):
                    check_disk_and_stop(args.output, guard)
                    path = directory / "rollouts" / f"{prompt['id']}__r{repeat+1:02d}.json"
                    if path.exists():
                        old = read_json(path)
                        if old["protocol_sha256"] != protocol["protocol_sha256"] or old["input_ids_sha256"] != input_hash:
                            raise RuntimeError("Existing rollout protocol/input mismatch")
                        state["completed_rollouts"] += 1
                        continue
                    state.update(status="generating", current_prompt=prompt["id"], current_rollout=repeat+1,
                                 updated_at=now())
                    recorder.save()
                    seed = seed_for(prompt["id"], repeat)
                    result = generate(model, inputs, observer, seed, protocol, guard, args.output, args.token_limit)
                    text = tokenizer.decode(result["token_ids"], skip_special_tokens=True)
                    initial_path = directory / "initials" / f"{prompt['id']}.npz"
                    if not initial_path.exists():
                        arrays = dict(observer.initial)
                        arrays.update({key+"_xi": project(value, basis, key) for key, value in observer.initial.items()})
                        write_npz(initial_path, **arrays)
                        write_json(initial_path.with_suffix(".json"), {"input_ids": input_ids[0].tolist(),
                                   "input_ids_sha256": input_hash, "protocol_sha256": protocol["protocol_sha256"],
                                   "basis_sha256": state["basis_sha256"], "basis_stage": f"{checkpoint}_nf4",
                                   "point": "full chat prefill, before any generated token including reasoning"})
                    with np.load(initial_path) as saved:
                        delta = max(float(np.linalg.norm(observer.initial[key] - saved[key])) for key in observer.initial)
                    trajectory = path.with_suffix(".trajectory.npz")
                    steps = np.array(observer.positions, dtype=np.int32)
                    write_npz(trajectory, generated_tokens_processed=steps,
                              processed_sequence_position=steps+int(input_ids.shape[1])-1,
                              xi_last=np.asarray(observer.path, dtype=np.float32))
                    result.update(text=text, prompt_id=prompt["id"], family=prompt["family"],
                                  group=prompt.get("group", prompt["id"]), rollout_index=repeat,
                                  checkpoint=checkpoint, precision=precision, seed=seed, created_at=now(),
                                  protocol_sha256=protocol["protocol_sha256"],
                                  basis_sha256=state["basis_sha256"], input_ids_sha256=input_hash,
                                  input_tokens=int(input_ids.shape[1]),
                                  initial_file=str(initial_path.relative_to(args.output)),
                                  trajectory_file=str(trajectory.relative_to(args.output)),
                                  initial_repeat_max_l2_difference=delta,
                                  scoring=inspect_response(prompt, text, result["truncated"]),
                                  smoke_token_limit=args.token_limit)
                    write_json(path, result)
                    state["completed_rollouts"] += 1
                    state.update(updated_at=now(), process_peak_rss_gib=guard.peak_rss/GIB,
                                 min_system_available_gib=guard.min_available/GIB,
                                 gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB)
                    recorder.save()
                    print(f"saved {prompt['id']} {repeat+1}/{rollouts} tokens={result['output_tokens']} "
                          f"finish={result['finish_reason']} seconds={result['total_seconds']:.1f}", flush=True)
                del inputs
            state.update(status="complete", finished_at=now())
            recorder.save()
        return 0
    except Exception:
        state.update(status="failed", finished_at=now(), error=traceback.format_exc())
        recorder.save()
        print(state["error"], flush=True)
        return 1
    finally:
        if observer:
            observer.close()
        for handle in bridges:
            handle.remove()
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def summarize(output):
    rows = []
    for stage in STAGES:
        for path in sorted((output / stage / "rollouts").glob("*.json")):
            row = read_json(path)
            rows.append({"stage": stage, "prompt": row["prompt_id"], "rollout": row["rollout_index"],
                         "tokens": row["output_tokens"], "seconds": row["total_seconds"],
                         "finish": row["finish_reason"], "rh_label": row["scoring"]["rh_label"],
                         "needs_review": row["scoring"]["review_required"]})
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output / "rollout_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Control RH: estado de adquisicion", "", f"Actualizado: {now()}", "",
             "Este resumen no valida la hipotesis predictiva. No hay un score de alineamiento ajustado.", "",
             "| Etapa | Respuestas guardadas | Truncadas | Etiquetas pendientes |",
             "|---|---:|---:|---:|"]
    for stage in STAGES:
        group = [r for r in rows if r["stage"] == stage]
        lines.append(f"| {stage} | {len(group)}/100 | {sum(r['finish'] != 'eos' for r in group)} "
                     f"| {sum(r['needs_review'] for r in group)} |")
    lines.extend(["", "Las tareas A/B del evaluador son un mismo grupo. Diez rollouts no son diez estados iniciales independientes.",
                  "", "Las coordenadas PCA se calibran sin etiquetas en 24 prompts publicos de entrenamiento, separados "
                  "de estos casos de control. No son una direccion semantica de alineamiento. Cada checkpoint usa su "
                  "propia base NF4 congelada para BF16; no se comparan sus ejes como si fueran identicos.",
                  "", "Las capturas iniciales contienen tres resumenes en bloques 6, 18 y 30. Las trayectorias "
                  "guardan xi del ultimo estado del bloque 18 cada ocho posiciones procesadas. Son exploratorias, "
                  "no entradas de un predictor previo a responder. Los NPZ son el registro cientifico; no se "
                  "mantuvo el visor SONAR abierto ni se modifico su codigo.",
                  "", "Se usa muestreo a temperatura 1, top-p 1, sin top-k, thinking desactivado y hasta 768 tokens. "
                  "EOS termina la respuesta; el limite se registra como truncamiento, nunca como ausencia de RH.",
                  "", "La frase y la eleccion de evaluador tienen endpoints automaticos limitados al protocolo del paper. "
                  "Chocolate, tareas de texto y codigo requieren revision semantica. No se ejecuta codigo generado.",
                  "", "La ficha publica del SFT no confirma que sea el checkpoint exacto del articulo. Este es un control "
                  "documentado en 8B, no una reproduccion exacta de los porcentajes publicados.", ""])
    (output / "ESTADO.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def stop_tree(process):
    import psutil
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=10)
        for own_process in alive:
            own_process.kill()
    except psutil.NoSuchProcess:
        pass


def controller(args):
    import psutil
    protocol = prepare(args.output)
    lock = args.output / "controller.lock.json"
    if lock.exists():
        previous = read_json(lock)
        try:
            existing = psutil.Process(previous["pid"])
            if abs(existing.create_time() - previous["process_created"]) < 1:
                raise RuntimeError("A controller is already running for this experiment")
        except psutil.NoSuchProcess:
            pass
    write_json(lock, {"pid": os.getpid(), "process_created": psutil.Process().create_time()})
    state = {"status": "preflight", "pid": os.getpid(), "started_at": now(),
             "budget_hours": args.hours, "protocol_sha256": protocol["protocol_sha256"]}
    status_path = args.output / "status.json"
    write_json(status_path, state)
    deadline = time.monotonic() + args.hours * 3600
    process = None
    try:
        validate_cached_models(args.output / "tokenizer_audit.json")
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        for stage in STAGES:
            stage_path = args.output / stage / "status.json"
            if stage_path.exists() and read_json(stage_path)["status"] == "complete":
                continue
            if (args.output / "STOP").exists():
                state["status"] = "stopped_by_user"
                break
            remaining = int(deadline - time.monotonic())
            if remaining < 60:
                state["status"] = "time_budget_reached"
                break
            state.update(status="running", stage=stage, updated_at=now())
            write_json(status_path, state)
            command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                       "worker", "--stage", stage, "--output", str(args.output),
                       "--max-seconds", str(remaining)]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with (args.output / f"{stage}.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=flags)
                state["worker_pid"] = process.pid
                write_json(status_path, state)
                while process.poll() is None:
                    if time.monotonic() >= deadline or (args.output / "STOP").exists():
                        stop_tree(process)
                        state["status"] = "stopped_by_user" if (args.output / "STOP").exists() else "time_budget_reached"
                        break
                    if shutil.disk_usage(args.output).free < 20 * GIB or psutil.virtual_memory().available < 3 * GIB:
                        stop_tree(process)
                        raise RuntimeError("Supervisor memory/disk safety stop")
                    try:
                        gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                             capture_output=True, text=True, timeout=10, creationflags=flags)
                        if gpu.returncode != 0:
                            raise RuntimeError("GPU monitoring command failed")
                        temperature = int(gpu.stdout.strip().splitlines()[0])
                        if temperature >= 82:
                            stop_tree(process)
                            raise RuntimeError(f"GPU temperature safety stop: {temperature} C")
                        state["gpu_temperature_c"] = temperature
                    except (ValueError, subprocess.TimeoutExpired) as exc:
                        stop_tree(process)
                        raise RuntimeError("GPU monitoring unavailable; stopping unattended job") from exc
                    state.update(updated_at=now(), free_ram_gib=psutil.virtual_memory().available/GIB,
                                 free_disk_gib=shutil.disk_usage(args.output).free/GIB)
                    write_json(status_path, state)
                    summarize(args.output)
                    time.sleep(15)
                process.wait(timeout=15)
                if state["status"] != "running":
                    break
                if process.returncode != 0:
                    raise RuntimeError(f"Stage {stage} failed with exit {process.returncode}; see its log")
        else:
            state["status"] = "complete"
        state["finished_at"] = now()
        write_json(status_path, state)
        summarize(args.output)
        return 0 if state["status"] == "complete" else 2
    except Exception:
        if process is not None and process.poll() is None:
            stop_tree(process)
        state.update(status="failed", finished_at=now(), error=traceback.format_exc())
        write_json(status_path, state)
        summarize(args.output)
        print(state["error"], flush=True)
        return 1
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if lock.exists() and read_json(lock).get("pid") == os.getpid():
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "run", "status", "stop", "summarize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=STAGES, default="original_nf4")
    parser.add_argument("--hours", type=float, default=23)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--rollout-limit", type=int)
    parser.add_argument("--token-limit", type=int)
    parser.add_argument("--calibration-limit", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not 0 < args.hours <= 24 or not 0 < args.max_seconds <= 86400:
        parser.error("Time budgets must be positive and at most 24 hours")
    if args.action == "worker":
        if any(value is not None for value in (args.prompt_limit, args.rollout_limit, args.token_limit, args.calibration_limit)) and args.output == DEFAULT_OUTPUT.resolve():
            parser.error("Smoke overrides require a separate output directory")
        return worker(args)
    if args.action == "run":
        return controller(args)
    if args.action == "prepare":
        protocol = prepare(args.output)
        print(json.dumps({"protocol_sha256": protocol["protocol_sha256"], "output": str(args.output),
                          "prompts": len(protocol["prompts"]), "calibration": len(protocol["calibration"])}))
    elif args.action == "stop":
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "STOP").touch()
        print("Stop requested; completed rollouts will be preserved.")
    elif args.action == "status":
        path = args.output / "status.json"
        print(json.dumps(read_json(path) if path.exists() else {"status": "not_started"}, indent=2))
    else:
        print("Saved rollouts:", len(summarize(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
