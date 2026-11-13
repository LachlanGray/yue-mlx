import json

import mlx.core as mx
import numpy as np
import pytest

from yue2_mlx.model import Config, AR, NAR
from yue2_mlx.protocol import prefix, chunk_ranges, Sampling
from yue2_mlx.pipeline import Pipeline


class Words:
    def encode(self, text):
        return [ord(c) for c in text]


def test_native_prefixes_and_chunk_limits():
    for cot in ("off", "full", "melody"):
        ids = prefix(Words(), "pop", "hello", cot, [65, 66] if cot != "off" else None)
        assert ids[0] == 151643 and ids[-1] == 151851
    for start, end in chunk_ranges(9000, 4096):
        assert 4096 + (end - start) * 2 + 3 <= 24576
    with pytest.raises(ValueError):
        chunk_ranges(1, 24576)
    with pytest.raises(ValueError):
        Sampling(min_tokens=20, max_tokens=10)


def test_no_cross_stage_modules():
    c = Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
               head_dim=8, intermediate_size=64, vocab_size=128)
    ar, nar = AR(c), NAR(c)
    assert not hasattr(ar.model.layers[0], "nar_mlp")
    assert not hasattr(nar.model, "embed_tokens")
    assert not hasattr(nar, "lm_head")


def test_huggingface_snapshot_symlinks(tmp_path):
    from yue2_mlx.weights import TensorSource
    blob = tmp_path / "blobs"
    snapshot = tmp_path / "snapshot"
    blob.mkdir()
    snapshot.mkdir()
    mx.save_safetensors(str(blob / "weights.safetensors"), {"weight": mx.ones((2, 3))})
    (snapshot / "model.safetensors").symlink_to(blob / "weights.safetensors")
    source = TensorSource(snapshot)
    np.testing.assert_array_equal(np.array(source.get("weight")), np.ones((2, 3)))


def test_stage_releases_materialized_weights(tmp_path):
    from mlx.utils import tree_flatten
    from dataclasses import asdict
    from yue2_mlx.weights import release_memory
    c = Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
               head_dim=16, intermediate_size=128, vocab_size=1024, max_position_embeddings=128)
    ar = AR(c)
    ar.set_dtype(mx.bfloat16)
    weights = dict(tree_flatten(ar.parameters()))
    mx.eval(weights)
    mx.save_safetensors(str(tmp_path / "ar.safetensors"), weights)
    manifest = {"format": "yue2-mlx-staged-v1", "model_config": asdict(c),
                "stages": {"ar": {"bits": 0, "group_size": 32, "files": ["ar.safetensors"], "keys": list(weights)}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    del weights, ar
    release_memory()
    baseline = mx.get_active_memory()
    pipe = Pipeline(tmp_path)
    with pipe._stage("ar") as model:
        logits, cache = model(mx.array([[1, 2, 3]]))
        mx.eval(logits, cache)
        assert mx.get_active_memory() > baseline + 100_000
        del logits, cache, model
    release_memory()
    assert mx.get_active_memory() < baseline + 32_768
    assert pipe.memory_events[-1]["stage"] == "ar"


def test_vae_only_round_trip_pipeline(reference, tmp_path):
    import soundfile as sf
    from test_parity import tiny_vae, export_vae
    from yue2_mlx.weights import convert
    ref, cfg = tiny_vae(reference)
    export_vae(ref, cfg, tmp_path / "source")
    manifest = convert("unused", str(tmp_path / "source"), tmp_path / "converted", vae_only=True, progress=None)
    assert set(manifest["stages"]) == {"encoder", "decoder"}
    t = np.arange(1920 * 4) / 48000
    signal = np.stack([.1 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    sf.write(tmp_path / "input.wav", signal, 48000, subtype="FLOAT")
    pipe = Pipeline(tmp_path / "converted", vae_core=2)
    pipe.reconstruct(tmp_path / "input.wav", tmp_path / "result")
    assert (tmp_path / "result/reconstruction.wav").exists()
    report = json.loads((tmp_path / "result/result.json").read_text())
    assert [e["stage"] for e in report["memory"]] == ["encoder", "decoder"]


def test_short_generation_all_stages(reference, tmp_path):
    """Real native vocabulary, tiny shared architecture, CFG, flow, saved audio."""
    from dataclasses import asdict
    from safetensors.torch import save_file
    import torch
    import soundfile as sf
    from yue2_mlx.weights import convert
    c = Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
               head_dim=8, intermediate_size=64, vocab_size=184704)
    torch.manual_seed(4)
    ref = reference.modeling_yue2.YuE2ForCausalLM(reference.modeling_yue2.YuE2Config(**asdict(c))).eval()
    source = tmp_path / "source"
    source.mkdir()
    save_file(ref.state_dict(), str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps(asdict(c)))
    common = dict(channels=2, c_mults=[1]*6, strides=[2, 2, 4, 4, 5, 6], use_snake=True)
    vc = reference.modeling_vae.YuE2VAEConfig(
        encoder_config=dict(common, in_channels=2, latent_dim=128),
        decoder_config=dict(common, out_channels=2, latent_dim=64, final_tanh=False))
    vae = reference.modeling_vae.YuE2VAE(vc)
    from test_parity import export_vae
    export_vae(vae, vc, tmp_path / "vae")
    convert(str(source), str(tmp_path / "vae"), tmp_path / "converted", bits=8, nar_bits=8, group_size=32, progress=None)
    del ref, vae
    pipe = Pipeline(tmp_path / "converted", vae_core=2, query_chunk=8)
    # Use a trivial tokenizer to avoid requiring a full downloaded BPE file.
    class Tokenizer:
        def encode(self, text):
            return [11, 12]
        def decode(self, ids):
            return "AB"
    pipe.tokenizer = Tokenizer()
    result = pipe.generate(tmp_path / "song", style="pop", lyrics="hi", cot="full", abc="AB",
                           sampling=Sampling(min_tokens=4, max_tokens=4, temperature=0), cfg_scale=1.2, steps=2)
    audio, rate = sf.read(result / "audio.flac")
    assert audio.shape == (4 * 1920 - 64, 2) and rate == 48000
    assert np.isfinite(audio).all()
    report = json.loads((result / "result.json").read_text())
    assert report["truncated"]
    assert [e["stage"] for e in report["memory"]] == ["ar", "ar", "nar", "decoder"]
