"""Oobleck encoder AND decoder, FP32, channels-last.

Architecture derived from YuE2 / stable-audio-tools; SnakeBeta from NVIDIA.
See THIRD_PARTY_NOTICES.md. Convolution weight normalization is folded offline.
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x):
        return x + mx.sin(x * mx.exp(self.alpha)) ** 2 / (mx.exp(self.beta) + 1e-9)


class Conv(nn.Conv1d):
    def __init__(self, cin, cout, kernel, stride=1, padding=0, dilation=1, bias=True):
        super().__init__(cin, cout, kernel, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self._geometry = (kernel, stride, padding, dilation, False)


class TransposeConv(nn.ConvTranspose1d):
    def __init__(self, cin, cout, stride):
        super().__init__(cin, cout, 2 * stride, stride=stride, padding=math.ceil(stride / 2))
        self._geometry = (2 * stride, stride, math.ceil(stride / 2), 1, True)


def activation(channels, snake):
    return SnakeBeta(channels) if snake else nn.ELU()


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation, snake=True):
        super().__init__()
        self.layers = [activation(channels, snake), Conv(channels, channels, 7, padding=3 * dilation, dilation=dilation),
                       activation(channels, snake), Conv(channels, channels, 1)]

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + y


class Block(nn.Module):
    def __init__(self, cin, cout, stride, snake=True, *, decode=False):
        super().__init__()
        if decode:
            self.layers = [activation(cin, snake), TransposeConv(cin, cout, stride)]
            self.layers += [ResidualUnit(cout, d, snake) for d in (1, 3, 9)]
        else:
            self.layers = [ResidualUnit(cin, d, snake) for d in (1, 3, 9)]
            self.layers += [activation(cin, snake), Conv(cin, cout, 2 * stride, stride, math.ceil(stride / 2))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def output_length(module, length):
    if hasattr(module, "_geometry"):
        k, s, p, d, transposed = module._geometry
        return ((length - 1) * s - 2 * p + d * (k - 1) + 1 if transposed
                else (length + 2 * p - d * (k - 1) - 1) // s + 1)
    if isinstance(module, ResidualUnit):
        return length
    if hasattr(module, "layers"):
        for layer in module.layers:
            length = output_length(layer, length)
    return length


def support(module, low, high):
    """Inclusive input indices needed for an output interval."""
    if hasattr(module, "_geometry"):
        k, s, p, d, transposed = module._geometry
        if transposed:
            return -(-(low + p - d * (k - 1)) // s), (high + p) // s
        return low * s - p, high * s - p + d * (k - 1)
    if hasattr(module, "layers"):
        a, b = low, high
        for layer in reversed(module.layers):
            a, b = support(layer, a, b)
        return (min(a, low), max(b, high)) if isinstance(module, ResidualUnit) else (a, b)
    return low, high


class AudioNetwork(nn.Module):
    def __init__(self, cfg, *, decode):
        super().__init__()
        if any(cfg.get(k, False) for k in ("antialias_activation", "use_nearest_upsample", "use_filter")):
            raise ValueError("Unsupported VAE activation/filter/upsampling option")
        if cfg.get("snake_type", "vanilla") != "vanilla":
            raise ValueError("Only vanilla SnakeBeta is supported")
        c = cfg.get("channels", 128)
        mults, strides = [1] + list(cfg.get("c_mults", [1, 2, 4, 8])), list(cfg.get("strides", [2, 4, 8, 8]))
        if len(mults) != len(strides) + 1 or min(strides) < 1:
            raise ValueError("Invalid VAE strides")
        snake = cfg.get("use_snake", False)
        self._ratio = math.prod(strides)
        self._decode = decode
        self._input_channels = cfg["latent_dim"] if decode else cfg.get("in_channels", 2)
        if decode:
            self.layers = [Conv(cfg["latent_dim"], c * mults[-1], 7, padding=3)]
            self.layers += [Block(c * mults[i], c * mults[i - 1], strides[i - 1], snake, decode=True)
                            for i in range(len(strides), 0, -1)]
            self.layers += [activation(c, snake), Conv(c, cfg.get("out_channels", 2), 7, padding=3, bias=False)]
            if cfg.get("final_tanh", True):
                self.layers.append(nn.Tanh())
        else:
            self.layers = [Conv(cfg.get("in_channels", 2), c, 7, padding=3)]
            self.layers += [Block(c * mults[i], c * mults[i + 1], s, snake) for i, s in enumerate(strides)]
            self.layers += [activation(c * mults[-1], snake), Conv(c * mults[-1], cfg["latent_dim"], 3, padding=1)]

    def __call__(self, x):
        if x.ndim != 3 or x.shape[-1] != self._input_channels or x.shape[1] < 1:
            raise ValueError("Expected nonempty [batch, time, channels] input")
        x = x.astype(mx.float32)
        for layer in self.layers:
            x = layer(x)
            mx.eval(x)
        return x

    def encode(self, audio, *, sample=False, seed=0):
        if self._decode:
            raise ValueError("Load the encoder for audio input")
        mean, scale = mx.split(self(audio), 2, axis=-1)
        if sample:
            stdev = nn.softplus(scale) + 1e-4
            mean = mean + mx.random.normal(mean.shape, key=mx.random.key(seed)) * stdev
        return mean

    def decode_tiled(self, latents, *, core=256, halo=None, progress=None):
        """Yield bounded NumPy waveform tiles; never collect GPU output tiles."""
        if not self._decode or type(core) is not int or core < 1:
            raise ValueError("Decoder and positive integer core required")
        z = np.asarray(latents, dtype=np.float32)
        if z.ndim != 2 or z.shape[1] != self._input_channels or len(z) < 1 or not np.isfinite(z).all():
            raise ValueError("Expected finite [frames, latent_dim] latents")
        low, high = support(self, 0, core * self._ratio - 1)
        required = max(0, -low, high - core + 1)
        halo = required if halo is None else halo
        if type(halo) is not int or halo < required:
            raise ValueError(f"halo must be >= {required}")
        total = output_length(self, len(z))
        for start in range(0, len(z), core):
            end = min(start + core, len(z))
            left, right = max(0, start - halo), min(len(z), end + halo)
            y = self(mx.array(z[None, left:right]))[0]
            count = min(end * self._ratio, total) - start * self._ratio
            crop = (start - left) * self._ratio
            result = np.array(y[crop:crop + count])
            del y
            if len(result) != count or not np.isfinite(result).all():
                raise FloatingPointError("Invalid decoded tile")
            if progress:
                progress(min(end, len(z)), len(z))
            yield result

    def encode_tiled(self, audio, *, core_frames=256, progress=None):
        """Deterministic posterior mean, aligned encoder tiles with exact halos."""
        if self._decode or type(core_frames) is not int or core_frames < 1:
            raise ValueError("Encoder and positive integer core_frames required")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != self._input_channels or len(audio) < self._ratio or not np.isfinite(audio).all():
            raise ValueError("Expected finite stereo audio with at least one latent frame")
        total = output_length(self, len(audio))
        for start in range(0, total, core_frames):
            end = min(start + core_frames, total)
            low, high = support(self, start, end - 1)
            left = max(0, (low // self._ratio) * self._ratio)
            right = min(len(audio), math.ceil((high + 1) / self._ratio) * self._ratio)
            z = self.encode(mx.array(audio[None, left:right]))[0]
            crop = start - left // self._ratio
            result = np.array(z[crop:crop + end - start])
            del z
            if len(result) != end - start or not np.isfinite(result).all():
                raise FloatingPointError("Invalid encoded tile")
            if progress:
                progress(end, total)
            yield result
