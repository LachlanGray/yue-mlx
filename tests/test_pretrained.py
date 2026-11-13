"""Opt-in real-weight parity. These tests never download checkpoints."""
import json

import mlx.core as mx
import numpy as np
import pytest
import torch

from yue2_mlx.weights import Checkpoint, release_memory


def test_pretrained_vae(reference, request, tmp_path):
    source = request.config.getoption("--vae-source")
    converted = request.config.getoption("--converted-vae")
    if not source or not converted:
        pytest.skip("Pass --vae-source and --converted-vae for pretrained parity")
    ref = reference.modeling_vae.YuE2VAE.from_pretrained(source)
    checkpoint = Checkpoint(converted)
    t = np.arange(48000, dtype=np.float32) / 48000
    clean = .1 * np.sin(2 * np.pi * 220 * t) + .04 * np.sin(2 * np.pi * 440 * t)
    rng = np.random.default_rng(7)
    audio = np.stack([clean + rng.normal(0, .01, len(t)), clean], axis=1).astype(np.float32)
    expected_z = ref.encode(torch.from_numpy(audio.T[None])).numpy()[0].T
    encoder = checkpoint.load("encoder")
    z = np.concatenate(list(encoder.encode_tiled(audio, core_frames=8)))
    del encoder
    release_memory()
    expected_audio = ref.decode(torch.from_numpy(expected_z.T[None])).numpy()[0].T
    decoder = checkpoint.load("decoder")
    # Decode identical latents for isolated decoder parity, then port-produced ones.
    decoded = np.concatenate(list(decoder.decode_tiled(expected_z, core=8)))
    roundtrip = np.concatenate(list(decoder.decode_tiled(z, core=8)))
    del decoder
    release_memory()
    def error(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-8))
    report = {"encoder_relative_l2": error(z, expected_z), "decoder_relative_l2": error(decoded, expected_audio),
              "roundtrip_relative_l2": error(roundtrip, expected_audio), "frames": len(z), "samples": len(decoded)}
    print("PRETRAINED_VAE " + json.dumps(report))
    assert report["encoder_relative_l2"] < .01
    assert report["decoder_relative_l2"] < .01
    assert report["roundtrip_relative_l2"] < .02
    import soundfile as sf
    sf.write(tmp_path / "input.wav", audio, 48000, subtype="FLOAT")
