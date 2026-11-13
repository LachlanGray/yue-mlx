"""Frozen NAR memory, positional layouts, strength, and one-way prefill."""
import math

import mlx.core as mx
import numpy as np
import pytest

from yue2_mlx.model import AR, NAR, Config, attention


def test_reference_attention_odds():
    q = mx.zeros((1, 1, 1, 4))
    k = mx.zeros((1, 1, 2, 4))
    v = mx.array([[[[0., 0., 0., 0.], [1., 1., 1., 1.]]]])
    result = attention(q, k, v, key_bias=mx.array([0., math.log(4)]))
    np.testing.assert_allclose(np.array(result), .8, atol=1e-6)


@pytest.mark.parametrize("placement", ["prefix", "audio"])
def test_frozen_reference_and_zero_bypass(placement, monkeypatch):
    mx.random.seed(23)
    c = Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=128,
               max_position_embeddings=128)
    ar, nar = AR(c), NAR(c)
    cache = ar.condition(mx.array([[1, 2, 3]]))
    source = mx.random.normal((4, 64))
    initial = mx.random.normal((5, 64))
    ref = nar.reference_cache(source, 0., cache, 3, 5, placement=placement)
    assert all(k.shape == (1, 2, 6, 8) for k, v in ref['kv'])
    before = [(np.array(k), np.array(v)) for k, v in ref['kv']]
    plain = nar.solve(initial, cache, 3, steps=2, start_time=.2)
    bypass = nar.solve(initial, cache, 3, steps=2, start_time=.2, reference=ref, reference_strength=0)
    np.testing.assert_array_equal(np.array(plain), np.array(bypass))
    # No reference prefill may occur during the solver.
    monkeypatch.setattr(nar, 'reference_cache', lambda *a, **kw: pytest.fail('Reference rebuilt during solve'))
    result = nar.solve(initial, cache, 3, steps=2, start_time=.2, reference=ref, reference_strength=1)
    assert result.shape == (5,64) and np.isfinite(np.array(result)).all()
    anchored = nar.solve(initial, cache, 3, steps=2, start_time=.2, reference=ref,
                         anchor_source=initial, anchor_noise=initial, anchor_strength=2)
    assert np.isfinite(np.array(anchored)).all()
    assert not np.array_equal(np.array(result), np.array(plain))
    for old, pair in zip(before, ref['kv']):
        for a,b in zip(old,pair):
            np.testing.assert_array_equal(a,np.array(b))
    # Same reference and prompt always give the same frozen memory, regardless
    # of target values: the builder takes only target length, not target state.
    other = NAR.reference_cache(nar, source, 0., cache, 3, 5, placement=placement)
    for old,pair in zip(before,other['kv']):
        for a,b in zip(old,pair):
            np.testing.assert_array_equal(a,np.array(b))
    with pytest.raises(ValueError):
        NAR.reference_cache(nar, source, 0., cache, 3, 123, placement=placement)


def test_layout_uses_different_positions():
    c = Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
               num_key_value_heads=2, head_dim=8, intermediate_size=64, vocab_size=128)
    ar, nar = AR(c), NAR(c)
    cache = ar.condition(mx.array([[1, 2]]))
    source = mx.ones((3,64))
    prefix = nar.reference_cache(source, 0., cache, 2, 5, placement='prefix')
    audio = nar.reference_cache(source, 0., cache, 2, 5, placement='audio')
    assert not np.array_equal(np.array(prefix['kv'][0][0]),np.array(audio['kv'][0][0]))
