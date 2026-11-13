"""One stage resident at a time, with actual release of unified-memory arrays."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
import soundfile as sf

from .protocol import Tokenizer, Sampling, ABC_SAMPLING, prefix, chunk_ranges, CODEC_OFFSET, CODEC_SIZE, MUSIC_END
from .sampling import generate
from .weights import Checkpoint, release_memory


class Pipeline:
    """Single-request staged inference. No model is resident between API calls.

    acoustic chunks reload AR conditioning weights then NAR weights. This trades
    disk reads for a smaller peak than keeping both networks in unified memory.
    """
    def __init__(self, model, *, memory_gib=10, cache_mib=256, query_chunk=256, vae_core=256, progress=None):
        if not math.isfinite(memory_gib) or memory_gib <= 0 or cache_mib < 0:
            raise ValueError("Invalid memory limits")
        if type(query_chunk) is not int or query_chunk < 1 or type(vae_core) is not int or vae_core < 1:
            raise ValueError("Chunk sizes must be positive integers")
        self.checkpoint = Checkpoint(model)
        self.query_chunk, self.vae_core = query_chunk, vae_core
        self.progress = progress
        self.memory_events = []
        # Allocator limits are process-wide; this pipeline is single-request.
        mx.set_memory_limit(int(memory_gib * 2**30))
        mx.set_cache_limit(int(cache_mib * 2**20))
        self.tokenizer = None
        self._active = None
        self.realaudio_adapter = None
        self.realaudio_adapter_scale = 1.
        self.ar_adapter = None
        self.ar_adapter_scale = 1.

    def _report(self, phase, done=0, total=0):
        if self.progress:
            self.progress(phase, done, total)

    @contextmanager
    def _stage(self, name, *, head=True):
        if self._active is not None:
            raise RuntimeError("A stage is already active; Pipeline is not concurrent")
        self._report("load_" + name)
        mx.reset_peak_memory()
        started = time.perf_counter()
        self._active = name
        model = None
        try:
            model = self.checkpoint.load(name, head=head)
            if name == "nar" and self.realaudio_adapter is not None:
                from .realaudio import apply_adapter
                apply_adapter(model, self.realaudio_adapter, self.realaudio_adapter_scale)
            if name == "ar" and self.ar_adapter is not None:
                from .realaudio import apply_ar_adapter
                apply_ar_adapter(model, self.ar_adapter, self.ar_adapter_scale)
            yield model
        finally:
            del model
            self._active = None
            release_memory()
            self.memory_events.append({"stage": name, "seconds": time.perf_counter() - started,
                                       "peak_gib": mx.get_peak_memory() / 2**30,
                                       "active_after_gib": mx.get_active_memory() / 2**30})

    def _tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(self.checkpoint.path / "qwen.tiktoken")
        return self.tokenizer

    def plan(self, *, style, lyrics, cot="full", abc=None, seed=831001, sampling=ABC_SAMPLING):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("MLX seed must be an integer in [0, 2**32)")
        if not isinstance(style, str) or not isinstance(lyrics, str):
            raise ValueError("style and lyrics must be strings")
        if abc is not None and (cot == "off" or not isinstance(abc, str) or not abc.strip()):
            raise ValueError("A supplied score requires nonempty ABC and full/melody mode")
        tokenizer = self._tokenizer()
        ids, truncated = None, False
        if cot != "off":
            if abc is not None:
                ids = tokenizer.encode(abc)
            else:
                with self._stage("ar") as model:
                    ids, truncated = generate(model, prefix(tokenizer, style, lyrics, cot), sampling, seed, "abc",
                                              query_chunk=self.query_chunk, progress=self._report)
                    del model
        tokens = prefix(tokenizer, style, lyrics, cot, ids)
        return {"style": style, "lyrics": lyrics, "cot": cot, "seed": seed, "abc_ids": ids,
                "abc": tokenizer.decode(ids) if ids is not None else None, "prefix": tokens,
                "truncated": truncated, "sampling": asdict(sampling)}

    def generate_semantic(self, plan, *, sampling=Sampling(), cfg_scale=None):
        cot = plan["cot"]
        scale = (1.01 if cot == "off" else 1.) if cfg_scale is None else cfg_scale
        if not math.isfinite(scale) or not scale >= 0:
            raise ValueError("cfg_scale must be finite and nonnegative")
        negative = prefix(self._tokenizer(), plan["style"], plan["lyrics"], cot, plan["abc_ids"], negative=True)
        with self._stage("ar") as model:
            ids, truncated = generate(model, plan["prefix"], sampling, plan["seed"], "semantic",
                                      negative=negative, cfg_scale=scale, legacy_off=cot == "off",
                                      query_chunk=self.query_chunk, progress=self._report)
            del model
        return {"plan": plan, "tokens": [i - CODEC_OFFSET for i in ids], "truncated": truncated,
                "sampling": asdict(sampling), "cfg_scale": scale}

    def semantic_audio(self, input_path, output_dir, *, tokenizer_model="models/realaudio-v4",
                       style="", lyrics="", steps=32, seed=831001, nar_adapter=True,
                       mode="fresh", start_time=1., noise_amount=None, anchor_strength=2.,
                       reference_path=None, reference_placement="prefix", reference_strength=1.):
        """Recorded semantic codes plus fresh/partial/source-guided acoustic flow."""
        from .realaudio import extract, package
        if type(steps) is not int or steps < 1 or type(nar_adapter) is not bool:
            raise ValueError("Use positive integer steps and a boolean NAR adapter option")
        if mode not in ("fresh", "mix", "direct", "anchored") or not math.isfinite(start_time) or not 0 <= start_time <= 1:
            raise ValueError("Use fresh/mix/direct/anchored and start time 0–1")
        if noise_amount is not None and (not math.isfinite(noise_amount) or not 0 <= noise_amount <= 1):
            raise ValueError("Noise amount must be 0–1 or automatic")
        if not math.isfinite(anchor_strength) or anchor_strength < 0:
            raise ValueError("Source pull must be finite and nonnegative")
        if reference_placement not in ("prefix", "audio") or not math.isfinite(reference_strength) or reference_strength < 0:
            raise ValueError("Invalid reference placement/strength")
        if self.checkpoint.manifest["model_config"].get("timestep_shift", 1.) != 1.:
            raise ValueError("Semantic editing requires timestep_shift=1")
        if mode == "fresh":
            start_time = 1.
        amount = 1. if mode == "fresh" else 0. if mode == "direct" else start_time if noise_amount is None else noise_amount
        if start_time == 0:
            amount = 0.  # Exact VAE baseline, regardless of custom noise setting.
        sources = package(tokenizer_model)
        plan = self.plan(style=style, lyrics=lyrics, cot="off", seed=seed)
        audio, rate = sf.read(input_path, dtype="float32", always_2d=True)
        if audio.shape[1] not in (1, 2) or len(audio) < rate / 25 or not np.isfinite(audio).all():
            raise ValueError("Choose finite mono/stereo audio with at least one 0.04-second acoustic frame")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        mx.reset_peak_memory()
        codes = extract(audio, rate, tokenizer_model, self._report)
        self.memory_events.append(dict(stage="audio_tokenizer", seconds=time.perf_counter()-started,
                                       peak_gib=mx.get_peak_memory()/2**30, active_after_gib=mx.get_active_memory()/2**30))
        semantic = dict(plan=plan, tokens=codes.tolist(), truncated=False)
        (output / "semantic.json").write_text(json.dumps(semantic, indent=2) + "\n")
        np.save(output / "semantic.npy", codes)
        # Save a matching-rate original and VAE-only baseline for comparison.
        from scipy.signal import resample_poly
        divisor = math.gcd(rate, 48000)
        audio = resample_poly(audio, 48000//divisor, rate//divisor, axis=0)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        sf.write(output / "input-resampled.wav", audio, 48000, subtype="FLOAT")
        baseline = self.encode(audio, 48000)
        del audio
        # MERT rounds frame counts; VAE convolution geometry may round down.
        # Crop the common interval, never stretch acoustic latents to fit tokens.
        n = min(len(codes), len(baseline))
        codes, baseline = codes[:n], baseline[:n]
        self.decode(baseline, output / "reconstruction.wav")
        semantic["tokens"] = codes.tolist()
        (output / "semantic.json").write_text(json.dumps(semantic, indent=2) + "\n")
        np.save(output / "semantic.npy", codes)
        np.save(output / "source-latents.npy", baseline)
        noise = np.array(mx.random.normal(baseline.shape, key=mx.random.key(seed)))
        initial = (1-amount)*baseline + amount*noise
        np.save(output / "initial-latents.npy", initial)
        reference_state = None
        if reference_path is not None:
            reference_audio = self._read_reference(reference_path)
            sf.write(output / "reference-resampled.wav", reference_audio, 48000, subtype="FLOAT")
            if reference_strength > 0 and start_time > 0:
                reference_state = self.encode(reference_audio, 48000)
                np.save(output / "reference-latents.npy", reference_state)
                ref_noise = np.array(mx.random.normal(reference_state.shape, key=mx.random.key(seed ^ 0x524546)))
                reference_state = (1-amount)*reference_state + amount*ref_noise
                np.save(output / "reference-initial-latents.npy", reference_state)
                del ref_noise
            del reference_audio
        previous = self.realaudio_adapter
        try:
            self.realaudio_adapter = tokenizer_model if nar_adapter else None
            latents = self.synthesize(semantic, steps=steps, initial=initial, start_time=start_time,
                                     anchor_source=baseline if mode == "anchored" else None,
                                     anchor_noise=noise, anchor_strength=anchor_strength,
                                     reference_state=reference_state, reference_placement=reference_placement,
                                     reference_strength=reference_strength)
        finally:
            self.realaudio_adapter = previous
        np.save(output / "latents.npy", latents)
        self.decode(latents, output / "semantic-audio.wav")
        report = dict(input=str(input_path), style=style, lyrics=lyrics, seed=seed, steps=steps,
                      mode=mode, start_time=start_time, noise_amount=noise_amount, effective_noise_amount=amount,
                      anchor_strength=anchor_strength if mode == "anchored" else 0,
                      reference_path=str(reference_path) if reference_path is not None else None,
                      reference_placement=reference_placement, reference_strength=reference_strength,
                      reference_policy="Frozen per acoustic chunk; sees semantic/text conditioning and reference, never target",
                      nar_adapter=nar_adapter, tokenizer_model=str(tokenizer_model),
                      tokenizer_sources=sources, semantic_frames=len(codes), memory=self.memory_events,
                      conditioning="Predicted real-audio semantic tokens plus optional source trajectory and frozen reference",
                      mert_precision="FP32", token_head_precision="FP32", channel_mode="mono semantics to stereo")
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def synthesize(self, semantic, *, steps=32, initial=None, start_time=1., anchor_source=None,
                   anchor_noise=None, anchor_strength=0., reference_state=None,
                   reference_placement="prefix", reference_strength=1.):
        tokens, plan = semantic["tokens"], semantic["plan"]
        if not math.isfinite(start_time) or not 0 <= start_time <= 1 or type(steps) is not int or steps < 1:
            raise ValueError("Use start time 0–1 and positive integer steps")
        if any(type(i) is not int or not 0 <= i < CODEC_SIZE for i in tokens):
            raise ValueError("Invalid semantic tokens")
        context = self.checkpoint.manifest["model_config"].get("max_position_embeddings", 24576)
        reference_frames = len(reference_state)+2 if reference_state is not None else 0
        ranges = chunk_ranges(len(tokens), len(plan["prefix"]), context=context-reference_frames)
        # Full-song noise drawn once before chunking. MLX RNG differs from torch.
        shape = (len(tokens), self.checkpoint.manifest["model_config"].get("latent_dim", 64))
        noise = np.array(mx.random.normal(shape, key=mx.random.key(plan["seed"]))) if initial is None else np.asarray(initial)
        if noise.shape != shape or not np.isfinite(noise).all():
            raise ValueError("Initial acoustic state must match semantic frames and be finite")
        if start_time == 0:
            return noise.copy()
        result = np.empty_like(noise)
        for index, (start, end) in enumerate(ranges):
            ids = plan["prefix"] + [CODEC_OFFSET + i for i in tokens[start:end]] + [MUSIC_END]
            self._report("acoustic_chunk", index + 1, len(ranges))
            cache = None
            try:
                with self._stage("ar", head=False) as model:
                    cache = model.condition(mx.array([ids]), query_chunk=self.query_chunk)
                    mx.eval(cache)
                    del model
                # Only the evaluated cache crosses the AR -> NAR boundary.
                with self._stage("nar") as model:
                    reference = None
                    if reference_state is not None:
                        self._report("reference_cache")
                        raw_t = max(-20., min(20., math.log(start_time/(1-start_time)))) if start_time < 1 else 20.
                        reference = model.reference_cache(mx.array(reference_state), raw_t, cache, len(ids), end-start,
                                                          placement=reference_placement, query_chunk=self.query_chunk)
                    anchor = dict(anchor_source=mx.array(anchor_source[start:end]),
                                  anchor_noise=mx.array(anchor_noise[start:end]), anchor_strength=anchor_strength) if anchor_source is not None and anchor_strength else {}
                    z = model.solve(mx.array(noise[start:end]), cache, len(ids), steps=steps,
                                    start_time=start_time, reference=reference, reference_strength=reference_strength, **anchor,
                                    query_chunk=self.query_chunk,
                                    progress=lambda d, t: self._report("flow", d, t))
                    result[start:end] = np.array(z)
                    del z, model, reference, anchor
            finally:
                del cache
                release_memory()
        return result

    def decode(self, latents, output):
        """Stream a waveform to a NEW file. Preserve the decoder's natural length."""
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        cfg = self.checkpoint.manifest["vae_config"]
        output.parent.mkdir(parents=True, exist_ok=True)
        with self._stage("decoder") as model:
            with sf.SoundFile(output, mode="w", samplerate=cfg["sample_rate"], channels=cfg["audio_channels"],
                              subtype="PCM_24") as f:
                for tile in model.decode_tiled(latents, core=self.vae_core,
                                               progress=lambda d, t: self._report("decode", d, t)):
                    f.write(np.clip(tile, -1, 1))
            del model
        return output

    def encode(self, audio, sample_rate):
        cfg = self.checkpoint.manifest["vae_config"]
        if sample_rate != cfg["sample_rate"]:
            raise ValueError(f"Resample input to {cfg['sample_rate']} Hz before encoding")
        with self._stage("encoder") as model:
            chunks = list(model.encode_tiled(audio, core_frames=self.vae_core,
                                             progress=lambda d, t: self._report("encode", d, t)))
            latents = np.concatenate(chunks)
            del model
        return latents

    def _independent_audio(self, input_path, output_dir, *, flow=False, **settings):
        """Two sequential dual-mono runs; average each output pair, preserve L/R order."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        audio, rate = sf.read(input_path, dtype="float32", always_2d=True)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        if audio.shape[1] != 2:
            raise ValueError("Only mono/stereo inputs are supported")
        target = self.checkpoint.manifest["vae_config"]["sample_rate"]
        if rate != target:
            from scipy.signal import resample_poly
            divisor = math.gcd(rate, target)
            audio = resample_poly(audio, target // divisor, rate // divisor, axis=0)
        sf.write(output / "input-resampled.wav", audio, target, subtype="FLOAT")
        reference_audio = None
        if flow and settings.get("reference_path") is not None:
            reference_audio = self._read_reference(settings["reference_path"])
            sf.write(output / "reference-resampled.wav", reference_audio, target, subtype="FLOAT")
        reports = []
        for index, channel in enumerate(("left", "right")):
            self._report("channel_" + channel)
            channel_input = output / (channel + "-input.wav")
            sf.write(channel_input, np.repeat(audio[:, index:index+1], 2, axis=1), target, subtype="FLOAT")
            folder = output / channel
            if flow:
                channel_settings = dict(settings)
                if reference_audio is not None:
                    ref_input = output / (channel + "-reference.wav")
                    sf.write(ref_input, np.repeat(reference_audio[:, index:index+1], 2, axis=1), target, subtype="FLOAT")
                    channel_settings["reference_path"] = ref_input
                self.flow_edit(channel_input, folder, **channel_settings)
            else:
                self.reconstruct(channel_input, folder)
            reports.append(json.loads((folder / "result.json").read_text()))
            # There is no single stereo latent array in this mode.
            for name in ("latents.npy", "source-latents.npy", "initial-latents.npy"):
                if (folder / name).is_file():
                    (folder / name).rename(output / (channel + "-" + name))
        for name in (["reconstruction.wav", "flow-edit.wav"] if flow else ["reconstruction.wav"]):
            with sf.SoundFile(output / "left" / name) as left, sf.SoundFile(output / "right" / name) as right:
                if left.frames != right.frames or left.samplerate != right.samplerate:
                    raise ValueError("Independent channel outputs must have identical length and sample rate")
                with sf.SoundFile(output / name, mode="w", samplerate=target, channels=2, subtype="PCM_24") as dest:
                    while True:
                        a, b = left.read(65536, always_2d=True), right.read(65536, always_2d=True)
                        if not len(a):
                            break
                        dest.write(np.stack((a.mean(axis=1), b.mean(axis=1)), axis=1))
        report = dict(reports[0])
        report.update(input=str(input_path), original_sample_rate=rate, input_samples=len(audio),
                      channel_mode="independent", channel_merge="mean of each dual-mono run; no normalization",
                      channel_reports={"left": "left/result.json", "right": "right/result.json"},
                      memory=self.memory_events)
        if flow:
            report["reference_path"] = str(settings["reference_path"]) if settings.get("reference_path") else None
            report["latent_rms_change_by_channel"] = dict(zip(("left", "right"), (r["latent_rms_change"] for r in reports)))
            report.pop("latent_rms_change", None)
            report["velocity_evaluations"] = sum(r["velocity_evaluations"] for r in reports)
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        return output

    def reconstruct(self, input_path, output_dir, *, independent_channels=False):
        """VAE round trip experiment, not a trained denoiser. Retains input and latents."""
        if independent_channels:
            return self._independent_audio(input_path, output_dir)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        audio, rate = sf.read(input_path, dtype="float32", always_2d=True)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        if audio.shape[1] != 2:
            raise ValueError("Only mono/stereo inputs are supported")
        target = self.checkpoint.manifest["vae_config"]["sample_rate"]
        if rate != target:
            from scipy.signal import resample_poly
            divisor = math.gcd(rate, target)
            audio = resample_poly(audio, target // divisor, rate // divisor, axis=0)
        sf.write(output / "input-resampled.wav", audio, target, subtype="FLOAT")
        z = self.encode(audio, target)
        np.save(output / "latents.npy", z)
        self.decode(z, output / "reconstruction.wav")
        report = {"experiment": "VAE posterior-mean round trip; not trained denoising",
                  "input": str(input_path), "original_sample_rate": rate, "sample_rate": target,
                  "input_samples": len(audio), "latent_frames": len(z), "memory": self.memory_events}
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        return output

    def _read_reference(self, path):
        info = sf.info(path)
        if info.channels not in (1, 2) or not info.duration >= .04:
            raise ValueError("Reference audio must be mono/stereo with at least one latent frame (0.04 seconds)")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if info.channels == 1:
            audio = np.repeat(audio, 2, axis=1)
        target = self.checkpoint.manifest["vae_config"]["sample_rate"]
        if rate != target:
            from scipy.signal import resample_poly
            divisor = math.gcd(rate, target)
            audio = resample_poly(audio, target // divisor, rate // divisor, axis=0)
        if not np.isfinite(audio).all():
            raise ValueError("Reference audio must be finite")
        return audio

    def flow_edit(self, input_path, output_dir, *, style="", lyrics="", start_time=.15,
                  mode="mix", steps=16, seed=831001, anchor_strength=2., independent_channels=False,
                  reference_path=None, reference_placement="prefix", reference_strength=1.):
        """Experimental text-only partial flow, not trained audio restoration.

        Mix assumes the straight interpolation z_t=(1-t)*z_source+t*epsilon.
        Direct intentionally feeds an unnoised encoding at a nonzero time.
        Anchored mode adds a time-scaled soft pull toward the source trajectory.
        No semantic tokens or inversion are used.
        """
        if mode not in ("mix", "direct", "anchored") or not math.isfinite(start_time) or not 0 <= start_time <= 1:
            raise ValueError("Use mix/direct/anchored and a start time in [0,1]")
        if not math.isfinite(anchor_strength) or not anchor_strength >= 0:
            raise ValueError("Source pull must be finite and nonnegative")
        if reference_placement not in ("prefix", "audio") or not math.isfinite(reference_strength) or not reference_strength >= 0:
            raise ValueError("Use prefix/audio reference placement and finite nonnegative strength")
        if type(steps) is not int or not steps >= 1 or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Use positive integer steps and a uint32 seed")
        if not isinstance(style, str) or not isinstance(lyrics, str):
            raise ValueError("Style and lyrics must be text")
        mc = self.checkpoint.manifest["model_config"]
        if mc.get("timestep_shift", 1.) != 1.:
            raise ValueError("Partial-flow experiments currently require timestep_shift=1")
        info = sf.info(input_path)
        if info.channels not in (1, 2) or not info.duration >= .04:
            raise ValueError("Flow lab requires mono/stereo audio with at least one latent frame (0.04 seconds)")
        if independent_channels:
            return self._independent_audio(input_path, output_dir, flow=True, style=style, lyrics=lyrics,
                                           start_time=start_time, mode=mode, steps=steps, seed=seed,
                                           anchor_strength=anchor_strength, reference_path=reference_path,
                                           reference_placement=reference_placement, reference_strength=reference_strength)
        reference_audio = self._read_reference(reference_path) if reference_path is not None else None
        # Reuse the exact preprocessing / posterior mean / baseline path.
        output = self.reconstruct(input_path, output_dir)
        source = np.load(output / "latents.npy", allow_pickle=False)
        (output / "latents.npy").rename(output / "source-latents.npy")
        reference_state = None
        if reference_audio is not None:
            sf.write(output / "reference-resampled.wav", reference_audio,
                     self.checkpoint.manifest["vae_config"]["sample_rate"], subtype="FLOAT")
            if reference_strength > 0 and start_time > 0:
                self._report("reference_encode")
                reference_state = self.encode(reference_audio, self.checkpoint.manifest["vae_config"]["sample_rate"])
                np.save(output / "reference-latents.npy", reference_state)
                if mode != "direct":
                    # Separate deterministic draw; target noise is unchanged.
                    ref_noise = np.array(mx.random.normal(reference_state.shape, key=mx.random.key(seed ^ 0x524546)))
                    reference_state = (1-start_time)*reference_state + start_time*ref_noise
                    del ref_noise
                np.save(output / "reference-initial-latents.npy", reference_state)
        del reference_audio
        ids = prefix(self._tokenizer(), style, lyrics, "off")
        # No semantic IDs or music-end token. NAR sees the text prefix only.
        if len(ids) + len(source) + 2 + (len(reference_state)+2 if reference_state is not None else 0) > mc.get("max_position_embeddings", 24576):
            raise ValueError("Text plus audio exceeds acoustic context; shorten the prompt or recording")
        initial = source.copy()
        noise = None
        if mode in ("mix", "anchored") and start_time > 0:
            noise = np.array(mx.random.normal(source.shape, key=mx.random.key(seed)))
            initial = (1 - start_time) * source + start_time * noise
            if mode != "anchored":
                noise = None
        np.save(output / "initial-latents.npy", initial)
        edited = initial
        if start_time > 0:
            cache = None
            try:
                with self._stage("ar", head=False) as model:
                    cache = model.condition(mx.array([ids]), query_chunk=self.query_chunk)
                    mx.eval(cache)
                    del model
                with self._stage("nar") as model:
                    frozen_reference = None
                    if reference_state is not None:
                        self._report("reference_cache")
                        raw_t = max(-20., min(20., math.log(start_time/(1-start_time)))) if start_time < 1 else 20.
                        frozen_reference = model.reference_cache(mx.array(reference_state), raw_t, cache, len(ids), len(source),
                                                                 placement=reference_placement, query_chunk=self.query_chunk)
                    anchor = dict(anchor_source=mx.array(source), anchor_noise=mx.array(noise),
                                  anchor_strength=anchor_strength) if mode == "anchored" and anchor_strength else {}
                    z = model.solve(mx.array(initial), cache, len(ids), steps=steps, start_time=start_time,
                                    query_chunk=self.query_chunk,
                                    progress=lambda d, t: self._report("flow", d, t),
                                    reference=frozen_reference, reference_strength=reference_strength, **anchor)
                    edited = np.array(z)
                    del z, model, anchor, frozen_reference
            finally:
                del cache
                release_memory()
        np.save(output / "latents.npy", edited)
        self.decode(edited, output / "flow-edit.wav")
        report = json.loads((output / "result.json").read_text())
        report.update(experiment="Experimental text-only partial flow; not trained denoising",
                      acoustic_checkpoint=str(self.realaudio_adapter) if self.realaudio_adapter is not None else None,
                      mode=mode, start_time=start_time, steps=steps, velocity_evaluations=2*steps if start_time else 0,
                      seed=seed, style=style, lyrics=lyrics, conditioning="text-only; no semantic audio tokens",
                      initialization="source unchanged" if mode == "direct" else "(1-t)*source+t*Gaussian",
                      anchor_strength=anchor_strength if mode == "anchored" else 0,
                      anchor_rule="post-step pull: 1-exp(-strength*dt); reference at new time; fixed initial noise" if mode == "anchored" else None,
                      reference_path=str(reference_path) if reference_path is not None else None,
                      reference_placement=reference_placement, reference_strength=reference_strength,
                      reference_active=reference_state is not None, reference_cache_time=start_time,
                      reference_policy="NAR K/V captured once; sees text and itself, never target; frozen during solve",
                      latent_rms_change=float(np.sqrt(np.mean((edited-source)**2))),
                      memory=self.memory_events)
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        return output

    def generate(self, output_dir, *, style, lyrics, cot="full", abc=None, seed=831001,
                 sampling=Sampling(), abc_sampling=ABC_SAMPLING, cfg_scale=None, steps=32):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        plan = self.plan(style=style, lyrics=lyrics, cot=cot, abc=abc, seed=seed, sampling=abc_sampling)
        (output / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        if plan["abc"] is not None:
            (output / "score.abc").write_text(plan["abc"])
        semantic = self.generate_semantic(plan, sampling=sampling, cfg_scale=cfg_scale)
        (output / "semantic.json").write_text(json.dumps(semantic, indent=2, ensure_ascii=False) + "\n")
        latents = self.synthesize(semantic, steps=steps)
        np.save(output / "latents.npy", latents)
        self.decode(latents, output / "audio.flac")
        report = {"truncated": plan["truncated"] or semantic["truncated"], "steps": steps,
                  "acoustic_checkpoint": str(self.realaudio_adapter) if self.realaudio_adapter is not None else None,
                  "query_chunk": self.query_chunk, "vae_core": self.vae_core,
                  "memory": self.memory_events, "sources": self.checkpoint.manifest["sources"]}
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        return output
