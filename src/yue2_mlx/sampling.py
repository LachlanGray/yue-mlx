"""Constrained native-vocabulary sampling with independent CFG caches."""
import mlx.core as mx
import numpy as np

from .protocol import EOD, ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE


def distribution(logits, sampling, history, step, phase, legacy_off=False):
    if phase not in ("abc", "semantic"):
        raise ValueError("Unknown generation phase")
    scores = logits if legacy_off else logits.astype(mx.float32)
    ids = mx.arange(scores.shape[-1])
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = ids < EOD if phase == "abc" else (ids >= CODEC_OFFSET) & (ids < CODEC_OFFSET + CODEC_SIZE)
    allowed = allowed | ((ids == end) & (step >= sampling.min_tokens))
    scores = mx.where(allowed, scores, -mx.inf)
    if history and sampling.repetition_penalty != 1:
        counts = np.zeros(scores.shape[-1], dtype=np.float32)
        np.add.at(counts, history[-sampling.penalty_window:], 1)
        alpha = mx.power(sampling.repetition_penalty, mx.array(counts).astype(scores.dtype))
        scores = mx.where(scores < 0, scores * alpha, scores / alpha)
    if sampling.temperature == 0:
        return scores
    scores = scores / sampling.temperature
    threshold = mx.sort(scores, axis=-1)[..., -min(sampling.top_k, scores.shape[-1]), None]
    scores = mx.where(scores >= threshold, scores, -mx.inf)
    if sampling.top_p < 1:
        order = mx.argsort(-scores, axis=-1)
        values = mx.take_along_axis(scores, order, axis=-1)
        probabilities = mx.softmax(values, axis=-1)
        removed = mx.cumsum(probabilities, axis=-1) - probabilities > sampling.top_p
        removed = removed & (mx.arange(scores.shape[-1]) >= (3 if legacy_off else 1))
        values = mx.where(removed, -mx.inf, values)
        scores = mx.take_along_axis(values, mx.argsort(order, axis=-1), axis=-1)
    return scores


def generate(model, tokens, sampling, seed, phase, *, negative=None, cfg_scale=1.,
             legacy_off=False, query_chunk=256, progress=None):
    context = model._config.max_position_embeddings
    if len(tokens) + sampling.max_tokens > context or not tokens:
        raise ValueError("Prefix + generation budget exceeds context, or prefix is empty")
    if cfg_scale != 1 and (negative is None or not negative or len(negative) + sampling.max_tokens > context):
        raise ValueError("CFG requires a negative prefix within context")
    def prefill(ids):
        cache, logits = None, None
        for start in range(0, len(ids), query_chunk):
            logits, cache = model(mx.array([ids[start:start + query_chunk]]), cache, query_chunk=query_chunk)
            mx.eval(logits, cache)
        return logits[:, -1, :], cache
    cond, cache = prefill(tokens)
    uncond, negative_cache = prefill(negative) if cfg_scale != 1 else (None, None)
    history, key = [], mx.random.key(seed)
    end = ABC_END if phase == "abc" else MUSIC_END
    for step in range(sampling.max_tokens):
        logits = cond if cfg_scale == 1 else uncond + cfg_scale * (cond - uncond)
        scores = distribution(logits, sampling, history, step, phase, legacy_off)
        key, draw_key = mx.random.split(key)
        if not mx.any(mx.isfinite(scores)).item():
            raise FloatingPointError("No finite sampling candidates")
        token = int((mx.argmax(scores, axis=-1) if sampling.temperature == 0 else
                     mx.random.categorical(scores, key=draw_key)).item())
        if progress:
            progress(phase, step + 1, sampling.max_tokens)
        if token == end:
            return history, False
        history.append(token)
        if step + 1 < sampling.max_tokens:
            cond, cache = model(mx.array([[token]]), cache, query_chunk=query_chunk)
            cond = cond[:, -1, :]
            if negative_cache is not None:
                uncond, negative_cache = model(mx.array([[token]]), negative_cache, query_chunk=query_chunk)
                uncond = uncond[:, -1, :]
            mx.eval(cond, cache, uncond, negative_cache)
    return history, True
