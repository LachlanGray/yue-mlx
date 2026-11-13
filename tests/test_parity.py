"""Consolidated shared-weight numerical checks against the upstream code.

Random inputs exercise the architecture, not trained music quality. FP32 checks
are tight; BF16 checks allow the expected backend-dependent rounding error.
"""
from dataclasses import asdict
import json

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from yue2_mlx.model import AR, NAR, Config, RMSNorm, rotary, attention
from yue2_mlx.vae import AudioNetwork, output_length
from yue2_mlx.weights import TensorSource, fold_conv, stage_keys, convert, Checkpoint


def arr(t):
    return mx.array(t.detach().float().numpy()).astype(mx.bfloat16 if t.dtype == torch.bfloat16 else mx.float32)


def close(a, b, *, atol=2e-5, rtol=2e-4):
    aa = np.array(a.astype(mx.float32)) if isinstance(a, mx.array) else np.asarray(a)
    bb = b.detach().float().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    np.testing.assert_allclose(aa, bb, atol=atol, rtol=rtol)


def setup_transformer(reference, dtype=torch.float32):
    c = Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
               head_dim=8, intermediate_size=64, vocab_size=128, max_position_embeddings=128,
               max_latent_frames=128, timestep_shift=1.7)
    torch.manual_seed(13)
    ref = reference.modeling_yue2.YuE2ForCausalLM(reference.modeling_yue2.YuE2Config(**asdict(c))).eval().to(dtype)
    state = ref.state_dict()
    ar, nar = AR(c), NAR(c)
    for model, stage in ((ar, "ar"), (nar, "nar")):
        model.load_weights([(k, arr(state[k])) for k in stage_keys(state, stage)])
    return c, ref, ar, nar


def test_norm_rotary_and_grouped_attention(reference):
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4, 8)
    norm = reference.modeling_yue2.RMSNorm(8)
    close(RMSNorm(8)(arr(x)), norm(x))
    positions = torch.arange(7, 10)[None]
    cos, sin = reference.modeling_yue2.RotaryEmbedding(8)(positions)
    expected = reference.modeling_yue2._apply_rotary(x, cos.unsqueeze(2), sin.unsqueeze(2))
    close(rotary(arr(x).transpose(0, 2, 1, 3), 7, 1e6).transpose(0, 2, 1, 3), expected)
    q, k, v = torch.randn(1, 4, 5, 8), torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8)
    mask = torch.arange(9)[None] <= torch.arange(4, 9)[:, None]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    close(attention(arr(q), arr(k), arr(v), causal=True, offset=4, query_chunk=2), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ar_full_incremental_and_chunked_prefill(reference, dtype):
    c, ref, ar, _ = setup_transformer(reference, dtype)
    ids = torch.tensor([[1, 9, 2, 31, 4, 19, 11]])
    with torch.no_grad():
        expected = ref(ids, use_cache=False).logits
    tol = dict(atol=3e-3, rtol=3e-2) if dtype == torch.bfloat16 else {}
    logits, _ = ar(mx.array(ids.numpy()), last_only=False, query_chunk=3)
    close(logits, expected, **tol)
    cache, outputs = None, []
    for start, end in ((0, 3), (3, 5), (5, 6), (6, 7)):
        logits, cache = ar(mx.array(ids[:, start:end].numpy()), cache, last_only=False, query_chunk=2)
        outputs.append(logits)
    close(mx.concatenate(outputs, axis=1), expected, **tol)
    assert cache[0][0].shape == (1, c.num_key_value_heads, 7, c.head_dim)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nar_condition_velocity_midpoint(reference, dtype):
    _, ref, ar, nar = setup_transformer(reference, dtype)
    ids = [1, 2, 8, 19, 4, 11]
    torch.manual_seed(29)
    noise = torch.randn(7, 64)
    chunk = reference.nar.Chunk(ids, noise)
    engine = reference.nar.CachedNAR(ref, chunk, attention="math", query_chunk_size=3)
    cache = ar.condition(mx.array([ids]), query_chunk=3)
    tol = dict(atol=4e-3, rtol=5e-2) if dtype == torch.bfloat16 else {}
    for (k, v), (rk, rv) in zip(cache, engine.cache):
        close(k[0].transpose(1, 0, 2), rk, **tol)
        close(v[0].transpose(1, 0, 2), rv, **tol)
    for raw_t in (-3., 0., 2., 20.):
        close(nar.velocity(arr(noise), raw_t, cache, len(ids), query_chunk=3),
              engine.velocity(noise.to(dtype), raw_t), **tol)
    result = nar.solve(arr(noise), cache, len(ids), steps=3, query_chunk=3)
    expected = engine.solve(steps=3)
    close(result, expected, **(dict(atol=.025, rtol=.05) if dtype == torch.bfloat16 else dict(atol=3e-5, rtol=2e-4)))


def tiny_vae(reference, final_tanh=False):
    common = dict(channels=2, c_mults=[1, 1, 1, 1, 1, 1], strides=[2, 2, 4, 4, 5, 6], use_snake=True)
    cfg = reference.modeling_vae.YuE2VAEConfig(
        encoder_config=dict(common, in_channels=2, latent_dim=4),
        decoder_config=dict(common, out_channels=2, latent_dim=2, snake_type="vanilla", final_tanh=final_tanh),
        latent_dim=2)
    torch.manual_seed(39)
    return reference.modeling_vae.YuE2VAE(cfg), cfg


def export_vae(ref, cfg, path):
    path.mkdir()
    save_file(ref.state_dict(), str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(cfg.to_dict()))


@pytest.mark.parametrize("final_tanh", [False, True])
def test_vae_encoder_decoder_and_tiling(reference, tmp_path, final_tanh):
    ref, cfg = tiny_vae(reference, final_tanh)
    export_vae(ref, cfg, tmp_path / "vae")
    source = TensorSource(tmp_path / "vae")
    encoder = AudioNetwork(cfg.encoder_config, decode=False)
    decoder = AudioNetwork(cfg.decoder_config, decode=True)
    for model, stage in ((encoder, "encoder"), (decoder, "decoder")):
        pairs = [fold_conv(k, source, stage) for k in stage_keys(source.tensors, stage)]
        model.load_weights([p for p in pairs if p is not None])
    torch.manual_seed(42)
    z = torch.randn(1, 2, 35) * .1
    expected = ref.decode(z)
    close(decoder(arr(z).transpose(0, 2, 1)).transpose(0, 2, 1), expected, atol=3e-5)
    tiled = np.concatenate(list(decoder.decode_tiled(z[0].T.numpy(), core=8)))
    close(tiled.T[None], expected, atol=3e-5)
    assert len(tiled) == 35 * 1920 - 64
    audio = torch.randn(1, 2, 1920 * 35 + 73) * .1
    expected = ref.encode(audio)
    close(encoder.encode(arr(audio).transpose(0, 2, 1)).transpose(0, 2, 1), expected, atol=3e-5)
    tiled = np.concatenate(list(encoder.encode_tiled(audio[0].T.numpy(), core_frames=8)))
    close(tiled.T[None], expected, atol=3e-5)
    with pytest.raises(ValueError, match="halo"):
        list(decoder.decode_tiled(z[0].T.numpy(), core=8, halo=0))


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_checkpoint_conversion_and_separate_loading(reference, tmp_path, bits):
    c, ref, _, _ = setup_transformer(reference, torch.bfloat16)
    model_dir, vae_dir = tmp_path / "source", tmp_path / "vae"
    model_dir.mkdir()
    state = ref.state_dict()
    # Two source shards exercise indexed loading, including BF16 tensors.
    keys = list(state)
    mapping = {}
    for index, names in enumerate((keys[::2], keys[1::2])):
        file = f"model-{index}.safetensors"
        save_file({k: state[k] for k in names}, str(model_dir / file))
        mapping.update({k: file for k in names})
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (model_dir / "config.json").write_text(json.dumps(asdict(c)))
    vae, vc = tiny_vae(reference)
    export_vae(vae, vc, vae_dir)
    output = tmp_path / "converted"
    manifest = convert(str(model_dir), str(vae_dir), output, bits=bits, nar_bits=bits, group_size=32, progress=None)
    checkpoint = Checkpoint(output)
    for stage in ("ar", "nar", "encoder", "decoder"):
        assert manifest["stages"][stage]["files"]
        loaded = checkpoint.load(stage)
        assert loaded is not None
        del loaded
    ar = checkpoint.load("ar")
    ids = torch.tensor([[1, 7, 14]])
    with torch.no_grad():
        expected = ref(ids).logits[:, -1:]
    actual, _ = ar(mx.array(ids.numpy()))
    close(actual, expected, atol=.02 if bits == 4 else .004, rtol=.1)
    conditioning = checkpoint.load("ar", head=False)
    assert not hasattr(conditioning, "lm_head")
    assert len(conditioning.condition(mx.array(ids.numpy()))) == c.num_hidden_layers
    with pytest.raises(FileExistsError):
        convert(str(model_dir), str(vae_dir), output)


def test_sampling_distribution_matches_reference(reference):
    from yue2_mlx.sampling import distribution
    from yue2_mlx.protocol import Sampling, CODEC_OFFSET, MUSIC_END
    torch.manual_seed(71)
    logits = torch.randn(1, 184704)
    settings = Sampling(top_k=40, top_p=.87)
    history = [CODEC_OFFSET + 8] * 3 + [CODEC_OFFSET + 9]
    for phase in ("abc", "semantic"):
        actual = distribution(arr(logits), settings, history, 300, phase)
        expected = reference.sampling.distribution(logits, settings, history, 300, phase)
        close(actual, expected)
    actual = distribution(arr(logits), settings, [], 0, "semantic")
    assert not np.isfinite(np.array(actual)[0, MUSIC_END])
