"""Partial-flow math and staged audio-edit orchestration (no reference checkout)."""
from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf

from yue2_mlx.model import Config, NAR
from yue2_mlx.pipeline import Pipeline


def test_partial_midpoint_times_and_replacement(monkeypatch):
    model = NAR(Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=8, intermediate_size=64))
    calls = []

    def velocity(state, raw_t, *args, **kwargs):
        calls.append((np.array(state), 1 / (1 + np.exp(-raw_t))))
        return mx.ones_like(state) * 2

    monkeypatch.setattr(model, "velocity", velocity)
    initial = mx.ones((5, 64))
    result = model.solve(initial, [], 3, steps=2, start_time=.2)
    np.testing.assert_allclose(np.array(result), .6, atol=1e-6)
    np.testing.assert_allclose([t for _, t in calls], [.2, .15, .1, .05], atol=1e-6)
    assert all(s.shape == (5, 64) for s, _ in calls)
    np.testing.assert_allclose(calls[2][0], .8, atol=1e-6)
    calls.clear()
    np.testing.assert_array_equal(np.array(model.solve(initial, [], 3, start_time=0)), np.array(initial))
    assert not calls
    for t in [-.1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            model.solve(initial, [], 3, start_time=t)


@pytest.mark.parametrize("mode,t", [("mix", .2), ("direct", .2), ("mix", 0), ("anchored", .2), ("anchored", 0)])
def test_edit_initialization_and_staging(tmp_path, mode, t):
    source = np.ones((25, 64), dtype=np.float32) * .3
    recording = tmp_path / "input.wav"
    sf.write(recording, np.zeros((48000, 2)), 48000)
    pipe = object.__new__(Pipeline)
    pipe.realaudio_adapter = None
    pipe.checkpoint = SimpleNamespace(manifest={"model_config": {"timestep_shift": 1}})
    pipe.query_chunk = 8
    pipe.memory_events = []
    pipe.progress = None
    pipe.tokenizer = SimpleNamespace(encode=lambda text: [1, 2])
    stages, seen = [], {}

    def reconstruct(path, output):
        output.mkdir()
        np.save(output / "latents.npy", source)
        (output / "result.json").write_text("{}")
        return output

    def solve(initial, cache, length, **kwargs):
        seen.update(initial=np.array(initial), kwargs=kwargs)
        return initial + .1

    @contextmanager
    def stage(name, **kwargs):
        stages.append(name)
        yield SimpleNamespace(condition=lambda ids, **kw: [], solve=solve)

    pipe.reconstruct = reconstruct
    pipe._stage = stage
    pipe.decode = lambda z, output: None
    output = pipe.flow_edit(recording, tmp_path / "result", mode=mode, start_time=t, seed=9)
    initial = np.load(output / "initial-latents.npy")
    if mode == "direct" or t == 0:
        np.testing.assert_array_equal(initial, source)
    else:
        expected = (1-t)*source + t*np.array(mx.random.normal(source.shape, key=mx.random.key(9)))
        np.testing.assert_array_equal(initial, expected)
    assert stages == (["ar", "nar"] if t else [])
    if t:
        assert seen["kwargs"]["start_time"] == t
        np.testing.assert_array_equal(seen["initial"], initial)
        if mode == "anchored":
            np.testing.assert_array_equal(np.array(seen["kwargs"]["anchor_source"]), source)
            np.testing.assert_array_equal(np.array(seen["kwargs"]["anchor_noise"]),
                                          np.array(mx.random.normal(source.shape, key=mx.random.key(9))))
    np.testing.assert_allclose(np.load(output / "latents.npy"), initial + (.1 if t else 0))
    np.testing.assert_array_equal(np.load(output / "source-latents.npy"), source)


def test_anchor_reference_time_and_zero_strength(monkeypatch):
    model = NAR(Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=8, intermediate_size=64))
    monkeypatch.setattr(model, "velocity", lambda state, *args, **kwargs: mx.ones_like(state))
    source, noise = mx.ones((3, 64))*2, mx.ones((3, 64))*4
    initial = .8*source + .2*noise
    plain = model.solve(initial, [], 3, steps=2, start_time=.2)
    zero = model.solve(initial, [], 3, steps=2, start_time=.2,
                       anchor_source=source, anchor_noise=noise, anchor_strength=0)
    np.testing.assert_array_equal(np.array(plain), np.array(zero))
    actual = model.solve(initial, [], 3, steps=2, start_time=.2,
                         anchor_source=source, anchor_noise=noise, anchor_strength=2)
    expected = np.array(initial)
    pull = -np.expm1(-2*.1)
    for next_t in [.1, 0]:
        expected -= .1
        expected += pull*((1-next_t)*np.array(source)+next_t*np.array(noise)-expected)
    np.testing.assert_allclose(np.array(actual), expected, atol=1e-6)
    # Constant reference: the total pull is independent of the number of steps.
    monkeypatch.setattr(model, "velocity", lambda state, *args, **kwargs: mx.zeros_like(state))
    for steps in [1, 4, 16]:
        actual = model.solve(mx.zeros((3, 64)), [], 3, steps=steps, start_time=.2,
                             anchor_source=source, anchor_noise=source, anchor_strength=2)
        np.testing.assert_allclose(np.array(actual), 2*(1-np.exp(-.4)), atol=1e-6)
    for strength in [-1, float("nan")]:
        with pytest.raises(ValueError):
            model.solve(initial, [], 3, anchor_strength=strength)
    with pytest.raises(ValueError):
        model.solve(initial, [], 3, anchor_strength=1)
