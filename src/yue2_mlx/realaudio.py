"""Mothersuperior v4 real-audio tokenizer. Inference only; no remote code execution.

MERT and the token head run separately. Native safetensors are read directly;
PyTorch is only needed by the optional parity tests, never by this runtime.
"""
from pathlib import Path
import json
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.signal import resample_poly

from .weights import TensorSource, release_memory

MERT_REPO = "m-a-p/MERT-v2-FullSong"
MERT_REV = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
TOKEN_REPO = "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
TOKEN_REV = "841258e9d245257e069d6b0ec8c476effff9b98c"
DEFAULT_ROOT = Path("models/realaudio-v4")


def install(output=DEFAULT_ROOT):
    """Download only pinned data files, then publish a local manifest last."""
    from huggingface_hub import snapshot_download
    root = Path(output).resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError(root / "manifest.json")
    root.mkdir(parents=True, exist_ok=True)
    cache = root.parent / "hf-cache"
    mert = snapshot_download(MERT_REPO, revision=MERT_REV, cache_dir=cache,
                             allow_patterns=["model.safetensors", "config.json", "LICENSE", "THIRD_PARTY_NOTICES.md"])
    head = snapshot_download(TOKEN_REPO, revision=TOKEN_REV, cache_dir=cache,
                             allow_patterns=["tokenizer_head_joint_v4.safetensors", "nar_lora_joint_v4.safetensors", "README.md"])
    manifest = dict(format="yue2-realaudio-v4", mert=str(Path(mert).resolve()),
                    tokenizer=str(Path(head).resolve()), mert_revision=MERT_REV, tokenizer_revision=TOKEN_REV)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def package(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    value = json.loads((root / "manifest.json").read_text())
    if value.get("format") != "yue2-realaudio-v4":
        raise ValueError("Expected a real-audio v4 package; run install-realaudio first")
    if value.get("base_tokenizer"):
        base = Path(value["base_tokenizer"])
        inherited = package(base if base.is_absolute() else root / base)
        value.setdefault("mert", inherited["mert"])
    for key in ("mert", "tokenizer"):
        path = Path(value[key])
        value[key] = str(path if path.is_absolute() else (root / path).resolve())
    return value


class Ops:
    """Named functional layers preserve the published checkpoint schema."""
    def __init__(self, weights):
        self.w = weights

    def linear(self, x, name):
        y = x @ self.w[name + ".weight"].T
        return y + self.w[name + ".bias"] if name + ".bias" in self.w else y

    def norm(self, x, name, eps=1e-5):
        return mx.fast.layer_norm(x, self.w[name + ".weight"], self.w[name + ".bias"], eps)

    def conv(self, x, name, *, padding=0, stride=1, groups=1):
        y = mx.conv1d(x, self.w[name + ".weight"].transpose(0, 2, 1),
                      padding=padding, stride=stride, groups=groups)
        return y + self.w[name + ".bias"] if name + ".bias" in self.w else y


class MERT(Ops):
    def mel(self, waveform):
        # Match torch STFT: centered reflect padding, periodic Hann, power,
        # HTK mel matrix from checkpoint, unclipped power dB, drop final frame.
        x = np.pad(np.asarray(waveform, np.float32), (1024, 1024), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(x, 2048)[::240]
        window = np.array(self.w["feature_extractor.spectrogram.window"])
        from scipy.fft import rfft
        power = np.abs(rfft(frames * window, axis=-1)) ** 2
        mel = mx.array(power) @ self.w["feature_extractor.mel_scale.fb"]
        mel = 10 * mx.log10(mx.maximum(mel, 1e-10))
        return ((mel[:-1] - self.w["feature_extractor.mel_mean"]) /
                mx.maximum(self.w["feature_extractor.mel_std"], 1e-5))[None]

    def convnext(self, x, name):
        h = self.conv(x, name + ".depthwise_block.1", padding=3, groups=x.shape[-1])
        p = name + ".pointwise_block"
        h = nn.gelu(self.linear(self.norm(h, p + ".0", 1e-6), p + ".1"))
        magnitude = mx.sqrt(mx.sum(h * h, axis=1, keepdims=True))
        normalized = magnitude / (mx.mean(magnitude, axis=-1, keepdims=True) + 1e-6)
        h = h + self.w[p + ".3.weight"] * h * normalized + self.w[p + ".3.bias"]
        return x + self.linear(h, p + ".4")

    def block(self, x, index):
        p = f"layers.{index}"
        def ff(h, name):
            return self.linear(nn.gelu(self.linear(h, name + ".w_1")), name + ".w_2")
        x = x + .5 * ff(self.norm(x, p + ".ffn1_layer_norm"), p + ".ffn1")
        h = self.norm(x, p + ".attn_layer_norm")
        q, k, v = [self.linear(h, p + ".attn." + n + "_proj").reshape(1, -1, 16, 64).transpose(0, 2, 1, 3)
                   for n in ("query", "key", "value")]
        q = mx.fast.rope(q, 64, traditional=False, base=10000., scale=1., offset=0)
        k = mx.fast.rope(k, 64, traditional=False, base=10000., scale=1., offset=0)
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=64 ** -.5)
        x = x + self.linear(h.transpose(0, 2, 1, 3).reshape(x.shape), p + ".attn.out_proj")
        c = p + ".conv_module"
        h = self.conv(self.norm(x, c + ".layer_norm"), c + ".conv_block.1")
        a, b = mx.split(h, 2, axis=-1)
        h = self.conv(a * mx.sigmoid(b), c + ".conv_block.3", padding=15, groups=x.shape[-1])
        h = nn.gelu(self.norm(h, c + ".conv_block.4.1"))
        x = x + self.conv(h, c + ".conv_block.6")
        x = x + .5 * ff(self.norm(x, p + ".ffn2_layer_norm"), p + ".ffn2")
        return self.norm(x, p + ".final_layer_norm")

    def __call__(self, waveform):
        if len(waveform) < 1025:
            raise ValueError("MERT needs at least 1025 samples at 24 kHz")
        x = self.mel(waveform)
        for i, depth in enumerate((3, 4, 5)):
            p = f"subsampling_module.{i}"
            if i:
                x = self.conv(self.norm(x, p + ".resampling_layer.0", 1e-6),
                              p + ".resampling_layer.2", stride=2)
            for j in range(depth):
                x = self.convnext(x, p + f".convnext_layers.{j}")
            mx.eval(x)
        for i in range(21):
            x = self.block(x, i)
            mx.eval(x)
        return x[0]


class TokenHead(Ops):
    def __call__(self, x):
        x = self.linear(x, "inp") + self.w["pos"][:, :x.shape[1]]
        width = x.shape[-1]
        for i in range(8):
            p = f"enc.layers.{i}"
            h = self.norm(x, p + ".norm1")
            h = h @ self.w[p + ".self_attn.in_proj_weight"].T + self.w[p + ".self_attn.in_proj_bias"]
            q, k, v = [a.reshape(1, -1, 8, width // 8).transpose(0, 2, 1, 3) for a in mx.split(h, 3, axis=-1)]
            h = mx.fast.scaled_dot_product_attention(q, k, v, scale=(width // 8) ** -.5)
            x = x + self.linear(h.transpose(0, 2, 1, 3).reshape(x.shape), p + ".self_attn.out_proj")
            x = x + self.linear(nn.gelu(self.linear(self.norm(x, p + ".norm2"), p + ".linear1")), p + ".linear2")
        return self.linear(self.norm(x, "norm"), "head")

    def predict(self, features):
        features = np.asarray(features, np.float32)
        x = (features - features.mean(0)) / (features.std(0) + 1e-5)
        total, win = len(x), 512
        if not total or not np.isfinite(x).all():
            raise ValueError("Empty or nonfinite MERT features")
        starts = list(range(0, max(1, total - win + 1), win // 2))
        if starts[-1] + win < total:
            starts.append(max(0, total - win))
        out = np.zeros(total, np.int32)
        for start in starts:
            n = min(win, total - start)
            window = np.pad(x[start:start+n], ((0, win-n), (0, 0)))
            pred = np.array(mx.argmax(self(mx.array(window[None])), axis=-1))[0, :n]
            lo = start + (0 if start == 0 else win // 4)
            hi = start + n - (0 if start+n >= total else win // 4)
            out[lo:hi] = pred[lo-start:hi-start]
        return out


def extract_features(audio, sample_rate, root=DEFAULT_ROOT, progress=lambda *a: None):
    """Return frame-aligned codes. Match v4's 30s MERT recipe, sequentially.

    Upstream drops subsecond tails before interpolating to full length. Preserve
    that behavior for parity; subsecond *whole clips* use a documented zero pad.
    """
    config = package(root)
    audio = np.asarray(audio, np.float32)
    mono = audio.mean(1) if audio.ndim == 2 else audio
    if not len(mono) or not np.isfinite(mono).all():
        raise ValueError("Audio must be nonempty and finite")
    divisor = math.gcd(sample_rate, 24000)
    mono = resample_poly(mono, 24000 // divisor, sample_rate // divisor)
    frames = max(1, round(len(mono) / 24000 * 25))
    short_clip = len(mono) < 24000
    if short_clip:
        mono = np.pad(mono, (0, 24000-len(mono)))
    chunks = [mono[s:s+720000] for s in range(0, len(mono), 720000) if len(mono[s:s+720000]) >= 24000]
    progress("load_mert", 0, 0)
    source = TensorSource(config["mert"])
    weights = {k: source.get(k) for k in source.tensors if not k.startswith("layers.") or int(k.split('.')[1]) <= 20}
    model = MERT(weights)
    features = []
    try:
        for i, chunk in enumerate(chunks):
            features.append(np.array(model(chunk)))
            progress("mert", i+1, len(chunks))
    finally:
        del model, weights, source
        release_memory()
    h = np.concatenate(features)
    if short_clip:
        # Keep the original interval, not a time-compressed padded second.
        h = h[:frames]
    # Torch interpolate linear, align_corners=False, then upstream FP16 storage.
    pos = np.clip((np.arange(frames) + .5) * len(h) / frames - .5, 0, len(h)-1)
    lo = np.floor(pos).astype(int)
    fraction = (pos-lo).astype(np.float32)[:, None]
    h = ((1-fraction)*h[lo] + fraction*h[np.minimum(lo+1, len(h)-1)]).astype(np.float16).astype(np.float32)
    return h


def extract(audio, sample_rate, root=DEFAULT_ROOT, progress=lambda *a: None):
    h = extract_features(audio, sample_rate, root, progress)
    config = package(root)
    progress("load_token_head", 0, 0)
    model = TokenHead(mx.load(str(Path(config["tokenizer"]) / "tokenizer_head_joint_v4.safetensors")))
    try:
        codes = model.predict(h)
    finally:
        del model
        release_memory()
    progress("tokenize_audio", len(codes), len(codes))
    return codes


class LoRALinear(nn.Module):
    def __init__(self, base, a, b, scale=1.):
        super().__init__()
        self.base, self.a, self.b, self.scale = base, a, b, scale

    def __call__(self, x):
        # Preserve quantized base weights; adapters are FP32, then cast like upstream.
        return self.base(x) + (self.scale * ((x.astype(mx.float32) @ self.a.T) @ self.b.T)).astype(x.dtype)


def apply_adapter(model, root=DEFAULT_ROOT, scale=1.):
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("NAR adapter scale must be finite and nonnegative")
    if scale == 0:
        return
    config = package(root)
    tensors = mx.load(str(Path(config["tokenizer"]) / "nar_lora_joint_v4.safetensors"))
    used = set()
    for i, layer in enumerate(model.model.layers):
        for group, names in (("nar_self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                             ("nar_mlp", ("gate_proj", "up_proj", "down_proj"))):
            module = getattr(layer, group)
            for name in names:
                key = f"layers.{i}.{group}.{name}"
                a, b = tensors[key + ".lora_A"], tensors[key + ".lora_B"]
                base = getattr(module, name)
                if b.shape[1] != a.shape[0] or b.shape[0] != base.weight.shape[0]:
                    raise ValueError("NAR adapter shape mismatch")
                setattr(module, name, LoRALinear(base, a, b, scale))
                used.update((key + ".lora_A", key + ".lora_B"))
    for name in ("vae2llm", "llm2vae"):
        module = getattr(model, name)
        for field in ("weight", "bias"):
            key = name + "." + field
            value = tensors[key]
            if value.shape != getattr(module, field).shape:
                raise ValueError("NAR adapter projection shape mismatch")
            base = getattr(module, field)
            setattr(module, field, (base.astype(mx.float32) + scale * (value.astype(mx.float32) - base.astype(mx.float32))).astype(module.weight.dtype))
            used.add(key)
    if used != set(tensors):
        raise ValueError("Unexpected adapter tensors")
    mx.eval(model.parameters())


def apply_ar_adapter(model, root, scale=1.):
    """Attach an inference-only AR LoRA exported by the upstream training scripts."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("AR adapter scale must be finite and nonnegative")
    if scale == 0:
        return
    config = json.loads((Path(root) / "manifest.json").read_text())
    if config.get("format") not in ("yue2-realaudio-v4", "yue2-ar-adapter"):
        raise ValueError("Expected a YuE2 real-audio package or AR adapter")
    filename = config.get("ar_adapter", "ar_lora.safetensors")
    tensor_root = Path(config.get("tokenizer", root))
    if not tensor_root.is_absolute():
        tensor_root = (Path(root).resolve() / tensor_root).resolve()
    tensors = mx.load(str(tensor_root / filename))
    used = set()
    for i, layer in enumerate(model.model.layers):
        for group, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                             ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            module = getattr(layer, group)
            for name in names:
                key = f"layers.{i}.{group}.{name}"
                a, b = tensors[key + ".lora_A"], tensors[key + ".lora_B"]
                base = getattr(module, name)
                # QuantizedLinear stores a packed second weight dimension, so
                # validate rank and output width here; the call validates input.
                if b.shape[1] != a.shape[0] or b.shape[0] != base.weight.shape[0]:
                    raise ValueError("AR adapter shape mismatch")
                setattr(module, name, LoRALinear(base, a, b, scale))
                used.update((key + ".lora_A", key + ".lora_B"))
    if used != set(tensors):
        raise ValueError("Unexpected AR adapter tensors")
    mx.eval(model.parameters())
