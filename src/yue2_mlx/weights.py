"""Bounded checkpoint conversion and stage loading, including BF16 without torch."""
from collections import defaultdict
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import struct

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from .model import AR, NAR, Config
from .vae import AudioNetwork


class TensorSource:
    """Read a single safetensors tensor at a time, including sharded sources."""
    def __init__(self, directory):
        self.directory = Path(directory)
        index = self.directory / "model.safetensors.index.json"
        if index.exists():
            self.mapping = json.loads(index.read_text())["weight_map"]
        else:
            self.mapping = None
        files = sorted(set(self.mapping.values())) if self.mapping is not None else ["model.safetensors"]
        self.tensors = {}
        for filename in files:
            path = self.directory / filename
            # HF snapshots intentionally symlink files into their blob store.
            # Validate the index filename, not the symlink's resolved parent.
            if Path(filename).name != filename:
                raise ValueError("Checkpoint shards must be directly inside the model directory")
            with path.open("rb") as f:
                length = struct.unpack("<Q", f.read(8))[0]
                if length > 100_000_000:
                    raise ValueError("Unreasonable safetensors header")
                header = json.loads(f.read(length))
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                if key in self.tensors:
                    raise ValueError(f"Duplicate tensor: {key}")
                self.tensors[key] = (path, 8 + length, spec)
        if self.mapping is not None and set(self.mapping) != set(self.tensors):
            raise ValueError("Shard index does not match tensors")

    def get(self, name):
        path, header, spec = self.tensors[name]
        types = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8", "I64": "<i8", "I32": "<i4"}
        dtype = np.dtype(types[spec["dtype"]])
        start, end = spec["data_offsets"]
        if end - start != int(np.prod(spec["shape"])) * dtype.itemsize or header + end > path.stat().st_size:
            raise ValueError(f"Invalid tensor extent: {name}")
        mapped = np.memmap(path, mode="r", dtype=dtype, offset=header + start, shape=tuple(spec["shape"]))
        result = mx.array(np.array(mapped))
        del mapped
        return result.view(mx.bfloat16) if spec["dtype"] == "BF16" else result


def resolve(source, *, vae=False, revision=None):
    if Path(source).is_dir():
        return Path(source)
    from huggingface_hub import snapshot_download
    patterns = ["config.json", "*.safetensors", "*.safetensors.index.json"]
    if not vae:
        patterns += ["qwen.tiktoken", "yue2_generation_config.json"]
    return Path(snapshot_download(str(source), revision=revision, allow_patterns=patterns))


def stage_keys(keys, stage):
    for key in keys:
        if stage in {"encoder", "decoder"}:
            if key.startswith(stage + "."):
                yield key
        elif key == "model.norm.weight":
            yield key
        elif stage == "ar":
            if key.startswith(("model.embed_tokens.", "lm_head.")) or (key.startswith("model.layers.") and ".nar_" not in key):
                yield key
        elif stage == "nar":
            if ".nar_" in key or key.startswith(("llm2vae.", "vae2llm.", "time_embedder.")):
                yield key


def fold_conv(key, source, stage):
    """Return destination key/value, or None for the folded scale parameter."""
    destination = key.removeprefix(stage + ".")
    if key.endswith("weight_g"):
        return None
    value = source.get(key).astype(mx.float32)
    if key.endswith("weight_v"):
        g = source.get(key[:-1] + "g").astype(mx.float32)
        value = value * g / mx.sqrt(mx.sum(value * value, axis=(1, 2), keepdims=True))
        # Only decoder block layers.1 is a transposed convolution.
        pieces = destination.split(".")
        transposed = stage == "decoder" and len(pieces) == 5 and pieces[2:4] == ["layers", "1"]
        value = value.transpose(1, 2, 0) if transposed else value.transpose(0, 2, 1)
        destination = destination[:-2]
    return destination, value


def quantizable(path):
    return path == "lm_head" or (path.startswith("model.layers.") and any(
        part in path for part in ("self_attn.", "mlp.")) and path.endswith(("_proj",)))


def release_memory():
    mx.synchronize()
    gc.collect()
    mx.clear_cache()


def convert(model, vae, output, *, bits=8, nar_bits=8, group_size=64, vae_only=False,
            model_revision=None, vae_revision=None, progress=print):
    """Write separate component shards. Never load a full source model.

    Output must not exist: failed conversion is kept for inspection; no implicit overwrite.
    bits=0 preserves BF16. The audio VAE always remains FP32.
    """
    if bits not in (0, 4, 8) or nar_bits not in (0, 4, 8) or group_size not in (32, 64, 128):
        raise ValueError("Invalid quantization settings")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    vae_dir = resolve(vae, vae=True, revision=vae_revision)
    model_dir = None if vae_only else resolve(model, revision=model_revision)
    vc = json.loads((vae_dir / "config.json").read_text())
    mc = None if vae_only else json.loads((model_dir / "config.json").read_text())
    if mc is not None:
        Config.from_dict(mc)
    sources = {"vae": TensorSource(vae_dir)}
    if model_dir is not None:
        sources["model"] = TensorSource(model_dir)
    output.mkdir(parents=True)
    manifest = {"format": "yue2-mlx-staged-v1", "model_config": mc, "vae_config": vc,
                "sources": {"model": str(model), "vae": str(vae), "model_revision": model_revision,
                            "vae_revision": vae_revision}, "stages": {}}
    for stage in (["encoder", "decoder"] if vae_only else ["ar", "nar", "encoder", "decoder"]):
        source = sources["model" if stage in ("ar", "nar") else "vae"]
        selected = list(stage_keys(source.tensors, stage))
        if not selected:
            raise ValueError(f"No source tensors for {stage}")
        groups = defaultdict(list)
        for key in selected:
            # Transformer layer shards ~50M parameters; VAE tensors individually.
            group = ".".join(key.split(".")[:3]) if key.startswith("model.layers.") else key.rsplit(".", 1)[0]
            groups[group].append(key)
        stage_bits = (bits if stage == "ar" else nar_bits) if stage in ("ar", "nar") else 0
        info = {"bits": stage_bits, "group_size": group_size, "files": [], "keys": [], "weight_bytes": 0}
        folder = output / stage
        folder.mkdir()
        for i, names in enumerate(groups.values()):
            tensors = {}
            for key in names:
                if stage in ("encoder", "decoder"):
                    pair = fold_conv(key, source, stage)
                    if pair is None:
                        continue
                    name, value = pair
                else:
                    name, value = key, source.get(key).astype(mx.bfloat16)
                if stage_bits and name.endswith(".weight") and quantizable(name[:-7]):
                    w, scales, biases = mx.quantize(value, group_size=group_size, bits=stage_bits)
                    tensors[name] = w
                    tensors[name[:-6] + "scales"] = scales
                    tensors[name[:-6] + "biases"] = biases
                else:
                    tensors[name] = value
                del value
            if not tensors:
                continue
            mx.eval(tensors)
            file = f"{stage}/{i:04d}.safetensors"
            mx.save_safetensors(str(output / file), tensors)
            info["files"].append(file)
            info["keys"].extend(tensors)
            info["weight_bytes"] += sum(v.nbytes for v in tensors.values())
            del tensors
            release_memory()
        manifest["stages"][stage] = info
        if progress:
            progress(f"Converted {stage}: {info['weight_bytes'] / 2**30:.2f} GiB")
    if model_dir is not None:
        for name in ("qwen.tiktoken", "yue2_generation_config.json"):
            if (model_dir / name).exists():
                shutil.copyfile(model_dir / name, output / name)
    # Manifest last: partial conversions cannot be loaded as completed packages.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format") != "yue2-mlx-staged-v1":
            raise ValueError("Expected a converted staged checkpoint")

    def load(self, stage, *, head=True):
        info = self.manifest["stages"][stage]
        if stage in ("ar", "nar"):
            c = Config.from_dict(self.manifest["model_config"])
            module = AR(c, head=head) if stage == "ar" else NAR(c)
            module.set_dtype(mx.bfloat16)
            if info["bits"]:
                nn.quantize(module, group_size=info["group_size"], bits=info["bits"],
                            class_predicate=lambda name, obj: isinstance(obj, nn.Linear) and quantizable(name))
        else:
            module = AudioNetwork(self.manifest["vae_config"][stage + "_config"], decode=stage == "decoder")
        expected = dict(tree_flatten(module.parameters()))
        names = {k for k in info["keys"] if head or not k.startswith("lm_head.")}
        if names != set(expected):
            raise ValueError(f"{stage} parameter schema mismatch: missing={set(expected)-names}, extra={names-set(expected)}")
        # Drop references to the lazily initialized parameters before replacing them.
        shapes = {k: v.shape for k, v in expected.items()}
        del expected
        loaded = set()
        for file in info["files"]:
            path = (self.path / file).resolve()
            if not path.is_relative_to(self.path.resolve()):
                raise ValueError("Weight shard escapes checkpoint directory")
            # Head-only shard can be skipped entirely during conditioning.
            if not head and stage == "ar" and all(k.startswith("lm_head.") for k in _file_keys(path)):
                continue
            tensors = mx.load(str(path))
            pairs = []
            for name, value in tensors.items():
                if not head and name.startswith("lm_head."):
                    continue
                if name not in shapes or name in loaded or value.shape != shapes[name]:
                    raise ValueError(f"Invalid or duplicate checkpoint tensor: {name}")
                pairs.append((name, value))
                loaded.add(name)
            module.load_weights(pairs, strict=False)
            mx.eval([value for _, value in pairs])
            del tensors, pairs
        if loaded != names:
            raise ValueError("Incomplete checkpoint")
        module.eval()
        return module


def _file_keys(path):
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return [k for k in json.loads(f.read(length)) if k != "__metadata__"]
