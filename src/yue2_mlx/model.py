"""Separately loadable AR and NAR networks. No PyTorch runtime dependency.

Equations and checkpoint names follow YuE2 (Apache-2.0, YuE2 authors).
Arrays use [batch, time, channel], attention caches [batch, head, time, dim].
"""
from dataclasses import dataclass, fields
import math

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Config:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 184704
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 24576
    latent_dim: int = 64
    max_latent_frames: int = 24576
    timestep_shift: float = 1.0

    def __post_init__(self):
        if any(getattr(self, f.name) <= 0 for f in fields(self)):
            raise ValueError("Model dimensions and numerical constants must be positive")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2 or self.hidden_size % 2:
            raise ValueError("Invalid grouped attention or sinusoidal dimensions")

    @classmethod
    def from_dict(cls, data):
        if data.get("tie_word_embeddings", False) or data.get("latent_type", "vae") != "vae":
            raise ValueError("Only the released untied VAE architecture is supported")
        return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        # Preserve the reference's FP32 reduction and low-precision multiply.
        inv = mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps)
        return x * inv.astype(x.dtype) * self.weight


def rotary(x, offset, theta):
    half = x.shape[-1] // 2
    freq = theta ** (-mx.arange(0, 2 * half, 2, dtype=mx.float32) / (2 * half))
    angles = mx.arange(offset, offset + x.shape[2], dtype=mx.float32)[:, None] * freq
    c, s = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return mx.concatenate((a * c - b * s, b * c + a * s), axis=-1)


def attention(q, k, v, *, causal=False, offset=0, query_chunk=256, key_bias=None):
    """Bound mask storage; retain the full visible key set for every query."""
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")
    pieces = []
    for start in range(0, q.shape[2], query_chunk):
        end = min(start + query_chunk, q.shape[2])
        kk, vv, mask = k, v, None
        if causal:
            visible = offset + end
            kk, vv = k[..., :visible, :], v[..., :visible, :]
            mask = mx.arange(visible)[None, :] <= mx.arange(offset + start, offset + end)[:, None]
        if key_bias is not None:
            bias = key_bias[:kk.shape[2]][None, :].astype(q.dtype)
            mask = bias if mask is None else mx.where(mask, bias, -float("inf"))
        y = mx.fast.scaled_dot_product_attention(q[..., start:end, :], kk, vv,
                                                scale=q.shape[-1] ** -0.5, mask=mask)
        mx.eval(y)
        pieces.append(y)
    return mx.concatenate(pieces, axis=2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self._config = c
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(c.head_dim, c.rms_norm_eps)

    def project(self, x, offset=0):
        c, (b, t, _) = self._config, x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, t, c.num_attention_heads, c.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(b, t, c.num_key_value_heads, c.head_dim))
        v = self.v_proj(x).reshape(b, t, c.num_key_value_heads, c.head_dim)
        return (rotary(q.transpose(0, 2, 1, 3), offset, c.rope_theta),
                rotary(k.transpose(0, 2, 1, 3), offset, c.rope_theta), v.transpose(0, 2, 1, 3))

    def output(self, q, k, v, **kwargs):
        y = attention(q, k, v, **kwargs).transpose(0, 2, 1, 3)
        return self.o_proj(y.reshape(y.shape[0], y.shape[1], -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class ARLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attn = Attention(c)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c)


class NARLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nar_input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.nar_self_attn = Attention(c)
        self.nar_pre_mlp_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.nar_mlp = MLP(c)


class Backbone(nn.Module):
    def __init__(self, c, nar=False):
        super().__init__()
        if not nar:
            self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = [(NARLayer if nar else ARLayer)(c) for _ in range(c.num_hidden_layers)]
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)


class AR(nn.Module):
    def __init__(self, config, *, head=True):
        super().__init__()
        self._config = config
        self.model = Backbone(config)
        if head:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, ids, cache=None, *, last_only=True, query_chunk=256):
        cache = [None] * len(self.model.layers) if cache is None else cache
        offset = 0 if cache[0] is None else cache[0][0].shape[2]
        if offset + ids.shape[1] > self._config.max_position_embeddings:
            raise ValueError("AR context exceeded")
        x = self.model.embed_tokens(ids)
        for i, layer in enumerate(self.model.layers):
            q, k, v = layer.self_attn.project(layer.input_layernorm(x), offset)
            if cache[i] is not None:
                k = mx.concatenate((cache[i][0], k), axis=2)
                v = mx.concatenate((cache[i][1], v), axis=2)
            cache[i] = k, v
            x = x + layer.self_attn.output(q, k, v, causal=True, offset=offset, query_chunk=query_chunk)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            mx.eval(x, k, v)
        x = self.model.norm(x[:, -1:] if last_only else x)
        return self.lm_head(x), cache

    def condition(self, ids, *, query_chunk=256):
        """Compute only per-layer AR keys/values; no vocabulary head required."""
        if ids.shape[1] > self._config.max_position_embeddings:
            raise ValueError("Conditioning context exceeded")
        x, cache = self.model.embed_tokens(ids), []
        for layer in self.model.layers:
            q, k, v = layer.self_attn.project(layer.input_layernorm(x))
            cache.append((k, v))
            x = x + layer.self_attn.output(q, k, v, causal=True, query_chunk=query_chunk)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            mx.eval(x, k, v)
        return cache


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = [nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)]

    def __call__(self, t, dtype):
        freq = mx.exp(-math.log(10000) * mx.arange(128, dtype=mx.float32) / 128)
        phase = t.astype(mx.float32).reshape(-1, 1) * freq
        x = mx.concatenate((mx.cos(phase), mx.sin(phase)), axis=-1).astype(dtype)
        for layer in self.mlp:
            x = layer(x)
        return x


def audio_positions(count, c, dtype):
    positions = mx.minimum(mx.arange(count, dtype=mx.float32), c.max_latent_frames - 1)
    freq = mx.exp(mx.arange(0, c.hidden_size, 2, dtype=mx.float32) * (-math.log(10000) / c.hidden_size))
    phase = positions[:, None] * freq
    return mx.stack((mx.sin(phase), mx.cos(phase)), axis=-1).reshape(count, c.hidden_size).astype(dtype)


class NAR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.model = Backbone(config, nar=True)
        self.vae2llm = nn.Linear(config.latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, config.latent_dim)
        self.time_embedder = TimestepEmbedder(config.hidden_size)

    def velocity(self, state, raw_t, cache, ar_length, *, query_chunk=256,
                 reference=None, reference_strength=1., position_offset=None, capture=False):
        c = self._config
        if state.ndim != 2 or state.shape[1] != c.latent_dim or state.shape[0] < 1:
            raise ValueError("Expected nonempty [frames, latent_dim] state")
        n = state.shape[0] + 2
        if not math.isfinite(reference_strength) or not reference_strength >= 0:
            raise ValueError("Reference strength must be finite and nonnegative")
        reference = reference if reference_strength else None
        ref_n = reference["kv"][0][0].shape[2] if reference else 0
        offset = ar_length + (ref_n if reference and reference["placement"] == "prefix" else 0)
        if position_offset is not None:
            offset = position_offset
        if max(ar_length+n+ref_n, offset+n) > c.max_position_embeddings or len(cache) != c.num_hidden_layers:
            raise ValueError("Invalid acoustic context or conditioning cache")
        dtype = self.vae2llm.weight.dtype
        t = mx.sigmoid(mx.array([raw_t], dtype=dtype))
        t = c.timestep_shift * t / (1 + (c.timestep_shift - 1) * t)
        x = self.vae2llm(mx.pad(state.astype(dtype), ((1, 1), (0, 0)))[None])
        x = x + self.time_embedder(t, dtype)[None]
        x = x + audio_positions(n, c, dtype)[None]
        captured = []
        for i, (layer, (ak, av)) in enumerate(zip(self.model.layers, cache)):
            q, k, v = layer.nar_self_attn.project(layer.nar_input_layernorm(x), offset)
            if capture:
                captured.append((k, v))
            bias = None
            if reference:
                rk, rv = reference["kv"][i]
                if reference["placement"] == "prefix":
                    keys, values = (ak, rk, k), (av, rv, v)
                    bias = mx.concatenate((mx.zeros((ak.shape[2],)), mx.full((ref_n,), math.log(reference_strength)), mx.zeros((n,))))
                else:
                    keys, values = (ak, k, rk), (av, v, rv)
                    bias = mx.concatenate((mx.zeros((ak.shape[2]+n,)), mx.full((ref_n,), math.log(reference_strength))))
            else:
                keys, values = (ak, k), (av, v)
            k, v = mx.concatenate(keys, axis=2), mx.concatenate(values, axis=2)
            x = x + layer.nar_self_attn.output(q, k, v, query_chunk=query_chunk, key_bias=bias)
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
            mx.eval(x)
        return captured if capture else self.llm2vae(self.model.norm(x))[0, 1:-1]

    def reference_cache(self, state, raw_t, cache, ar_length, target_frames, *, placement="prefix", query_chunk=256):
        """One-way reference prefill: sees AR + itself, never the target.

        Prefix: [AR, reference, target]. Acoustic: [AR, target, reference].
        Each audio block keeps its own local sinusoidal positions; RoPE uses
        the full sequence positions above. Both reference boundary frames stay.
        """
        if placement not in ("prefix", "audio") or target_frames < 1:
            raise ValueError("Invalid reference placement or target frame count")
        if ar_length + target_frames + state.shape[0] + 4 > self._config.max_position_embeddings:
            raise ValueError("Reference plus target exceeds acoustic context")
        offset = ar_length if placement == "prefix" else ar_length + target_frames + 2
        kv = self.velocity(state, raw_t, cache, ar_length, query_chunk=query_chunk,
                           position_offset=offset, capture=True)
        mx.eval(kv)
        return {"kv": kv, "placement": placement}

    def solve(self, noise, cache, ar_length, *, steps=32, query_chunk=256, progress=None, start_time=1.0,
              anchor_source=None, anchor_noise=None, anchor_strength=0., reference=None, reference_strength=1.):
        """Integrate an initial state from start_time to zero; replace, never append."""
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(start_time) or not 0 <= start_time <= 1:
            raise ValueError("start_time must be finite and in [0,1]")
        if not math.isfinite(reference_strength) or not reference_strength >= 0:
            raise ValueError("Reference strength must be finite and nonnegative")
        if not math.isfinite(anchor_strength) or not anchor_strength >= 0:
            raise ValueError("anchor_strength must be finite and nonnegative")
        if anchor_strength:
            if anchor_source is None or anchor_noise is None or anchor_source.shape != noise.shape or anchor_noise.shape != noise.shape:
                raise ValueError("Anchoring requires source and fixed noise matching the initial state shape")
            if not mx.all(mx.isfinite(anchor_source)).item() or not mx.all(mx.isfinite(anchor_noise)).item():
                raise ValueError("Anchor arrays must be finite")
        if start_time == 0:
            return noise.astype(mx.float32)
        state = noise.astype(self.vae2llm.weight.dtype)
        dt = start_time / steps
        pull = -math.expm1(-anchor_strength * dt)
        def raw(t):
            return max(-20., min(20., math.log(t / (1 - t)))) if 0 < t < 1 else (20. if t >= 1 else -20.)
        for i in range(steps):
            t = start_time - i * dt
            first = self.velocity(state, raw(t), cache, ar_length, query_chunk=query_chunk,
                                  reference=reference, reference_strength=reference_strength)
            mid = state - first * (dt / 2)
            state = state - self.velocity(mid, raw(t - dt / 2), cache, ar_length, query_chunk=query_chunk,
                                          reference=reference, reference_strength=reference_strength) * dt
            if anchor_strength:
                # Soft source-trajectory pull at the NEW time. Fixed noise is
                # reused, not redrawn or added independently at each step.
                next_t = max(0., start_time - (i + 1) * dt)
                source_reference = (1-next_t) * anchor_source + next_t * anchor_noise
                state = (state.astype(mx.float32) + pull * (source_reference - state)).astype(state.dtype)
            mx.eval(state)
            if progress:
                progress(i + 1, steps)
        if not mx.all(mx.isfinite(state)).item():
            raise FloatingPointError("Non-finite acoustic latents")
        return state.astype(mx.float32)
