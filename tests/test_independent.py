"""Channel isolation, ordering, merge gain and identical flow settings."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from yue2_mlx.pipeline import Pipeline


@pytest.mark.parametrize("flow", [False, True])
@pytest.mark.parametrize("mono", [False, True])
def test_independent_wrapper(tmp_path, flow, mono):
    pipe = object.__new__(Pipeline)
    pipe.checkpoint = SimpleNamespace(manifest={"vae_config": {"sample_rate": 48000}})
    pipe.progress = None
    pipe.memory_events = []
    audio = np.tile([.1, .3], (4800, 1)).astype(np.float32)
    if mono:
        audio = audio[:, :1]
    src = tmp_path / "input.wav"
    sf.write(src, audio, 48000, subtype="FLOAT")
    calls = []

    def process(path, folder, **kwargs):
        signal, rate = sf.read(path, always_2d=True)
        np.testing.assert_array_equal(signal[:, 0], signal[:, 1])
        calls.append((signal[0, 0], kwargs))
        folder.mkdir()
        # Deliberately unequal output pair: merge should take the mean,
        # preserving gain rather than normalizing or selecting one channel.
        result = signal * [1, 3]
        sf.write(folder / "reconstruction.wav", result, rate, subtype="FLOAT")
        if flow:
            sf.write(folder / "flow-edit.wav", result, rate, subtype="FLOAT")
        np.save(folder / "latents.npy", np.ones((2, 64)))
        (folder / "result.json").write_text(json.dumps({"latent_rms_change": .1, "velocity_evaluations": 4}))
        return folder

    pipe.reconstruct = pipe.flow_edit = process
    output = pipe._independent_audio(src, tmp_path / "output", flow=flow, seed=123, mode="anchored")
    result, rate = sf.read(output / ("flow-edit.wav" if flow else "reconstruction.wav"))
    expected = np.repeat(audio, 2, axis=1) if mono else audio
    np.testing.assert_allclose(result, expected * 2, atol=2e-7)
    assert result.shape == (4800, 2) and rate == 48000
    assert len(calls) == 2 and calls[0][1] == calls[1][1]
    if flow:
        assert calls[0][1]["seed"] == 123
    assert (output / "left-latents.npy").exists() and (output / "right-latents.npy").exists()
    report = json.loads((output / "result.json").read_text())
    assert report["channel_mode"] == "independent"
    if flow:
        assert report["velocity_evaluations"] == 8
