"""Convert upstream PyTorch tokenizer/NAR/AR training outputs for MLX inference."""
from pathlib import Path
import json

import numpy as np
from safetensors.numpy import save_file


PROJECTIONS = (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
               ("mlp", ("gate_proj", "up_proj", "down_proj")))


def _numpy(tensor):
    return tensor.detach().cpu().float().numpy().astype(np.float32, copy=False)


def _lora_names(values, *, nar=False):
    expected = 28 * 7 * 2
    if len(values) != expected:
        raise ValueError(f"Expected {expected} LoRA tensors, found {len(values)}")
    result, cursor = {}, 0
    for layer in range(28):
        for group, names in PROJECTIONS:
            group = "nar_" + group if nar else group
            for name in names:
                a, b = values[cursor:cursor + 2]; cursor += 2
                if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                    raise ValueError("Malformed LoRA tensor pair")
                key = f"layers.{layer}.{group}.{name}"
                result[key + ".lora_A"] = _numpy(a)
                result[key + ".lora_B"] = _numpy(b)
    return result


def import_ar_checkpoint(source, output, name=None):
    """Convert one upstream AR step/best/last checkpoint without duplicating NAR data."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Importing .pt training outputs requires the optional test dependencies") from exc
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    ar = torch.load(source, map_location="cpu", weights_only=True)
    ar_state = _lora_names(ar.get("lora", []), nar=False)
    output.mkdir(parents=True, exist_ok=True)
    save_file(ar_state, str(output / "ar_lora.safetensors"),
              metadata={"rank": str(ar.get("rank", ar["lora"][0].shape[0])),
                        "step": str(ar.get("step", source.stem.removeprefix("step-")))})
    manifest = dict(format="yue2-ar-adapter", name=name or source.parent.parent.name.replace("_", " "),
                    ar_adapter="ar_lora.safetensors",
                    source_checkpoint=str(Path("..") / source.parent.name / source.name),
                    step=ar.get("step"), rank=ar.get("rank"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def import_training_bundle(source, output, base_tokenizer="models/realaudio-v4"):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Importing .pt training outputs requires the optional test dependencies") from exc
    source, output, base = Path(source).resolve(), Path(output).resolve(), Path(base_tokenizer).resolve()
    required = (source / "joint/head_best.pt", source / "joint/lora_best.pt", source / "ar/step-1300.pt")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Expected joint/head_best.pt, joint/lora_best.pt, and ar/step-1300.pt")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    base_manifest = json.loads((base / "manifest.json").read_text())
    if base_manifest.get("format") != "yue2-realaudio-v4":
        raise ValueError("Base tokenizer is not a YuE2 real-audio package")
    head = torch.load(required[0], map_location="cpu", weights_only=True)
    nar = torch.load(required[1], map_location="cpu", weights_only=True)
    ar = torch.load(required[2], map_location="cpu", weights_only=True)
    head_state = head.get("model")
    if not isinstance(head_state, dict) or "pos" not in head_state:
        raise ValueError("Unrecognized tokenizer head checkpoint")
    nar_state = _lora_names(nar.get("lora", []), nar=True)
    for module in ("vae2llm", "llm2vae"):
        for field in ("weight", "bias"):
            nar_state[f"{module}.{field}"] = _numpy(nar["io"][module][field])
    ar_state = _lora_names(ar.get("lora", []), nar=False)
    output.mkdir(parents=True, exist_ok=True)
    save_file({k: _numpy(v) for k, v in head_state.items()}, str(output / "tokenizer_head_joint_v4.safetensors"))
    save_file(nar_state, str(output / "nar_lora_joint_v4.safetensors"), metadata={"rank": str(nar.get("rank", 32))})
    save_file(ar_state, str(output / "ar_lora.safetensors"), metadata={"rank": str(ar.get("rank", 64)), "step": str(ar.get("step", 1300))})
    manifest = dict(format="yue2-realaudio-v4", name=source.name.replace("_", " "),
                    base_tokenizer=str(Path("..") / ".." / ".." / base.name), tokenizer=".",
                    ar_adapter="ar_lora.safetensors", source_bundle="..",
                    tokenizer_revision="local-training-bundle",
                    mert_revision=base_manifest.get("mert_revision"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
