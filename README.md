# YuE2 MLX

Independent MLX implementation of **YuE2-3B**, including its audio VAE encoder
and decoder. Uses the original Hugging Face weights. PyTorch and the upstream
checkout are only needed for optional reference tests, never for inference or
conversion. This is not the community `ahmadw` implementation.

The pipeline supports song generation, supplied ABC scores, native semantic CFG,
midpoint acoustic flow matching, standalone encoding/decoding, and an experimental
audio VAE round trip. A round trip is **not a trained denoiser**: it may remove noise,
retain it, or damage musical detail. No audio-cleaning quality claim is made.

## Architecture guide

Open [context/index.html](context/index.html) in a browser for an offline guide to
the model's representations, loading boundaries, and flow-matching inference.
The [architecture page](context/architecture.html) includes rendered SVG diagrams
and their editable D2 sources.

## Semantic lab: recording → tokens → audio

The **Semantic lab** tab uses Mothersuperior's real-audio tokenizer v4 to
predict YuE2 semantic codes from an uploaded recording. It then runs the existing
AR conditioning pass, NAR flow matching and VAE decoder. It does not generate
semantic tokens autoregressively, nor invert the source's acoustic trajectory.
The acoustic state can start from fresh Gaussian noise or source-guided partial
flow. Original audio, a VAE-only
baseline, semantic reconstruction, tokens and settings are saved for comparison.
Jobs use the same persistent queue and **Use settings** history controls.

The main workspace mounts audio players for active/queued jobs and only three
recent results. **Browse full history** opens `/history`, six finished sessions
per page. Leaving a page pauses and detaches its players; historical records and
audio files are retained. **Use settings** returns to the workspace with the
chosen inputs/settings restored, without submitting a job.

Semantic lab's **Acoustic method** combines the extracted song tokens with:

- **Fresh noise:** original full resynthesis, fixed start time 1.
- **Noise + resume:** `(1-a)*source + a*noise`, integrating from the chosen t to 0.
- **Direct refinement:** unnoised source encoding, integrating from chosen t.
- **Anchored:** noise + resume with the existing soft source-trajectory pull.

Noise amount `a` defaults to t, or can be independently set from 0 to 1.
Mismatched noise amount/time is an experimental initialization, not an inverse
diffusion procedure. The anchor still follows `(1-current_t)*source + current_t*noise`
using the same Gaussian draw, even with a custom initial amount. Start time zero
returns the aligned VAE baseline and bypasses AR/NAR. Source latents, initial
latents, effective noise amount and all settings are saved.

Optional reference audio supports both existing frozen-cache placements and
reference strength. Reference initialization uses the same effective noise mix
amount; it is clean in direct mode and pure noise in fresh mode. Its cache is
captured once per acoustic chunk using that chunk's semantic/text conditioning,
then remains fixed throughout the solve. Reference frames reduce available
chunk context; the target is chunked accordingly, with no new duration cap.
MERT/VAE frame-count differences are handled by cropping to their common interval,
not stretching latent values. This combined mode still uses joint stereo processing.

```sh
.venv/bin/python -m yue2_mlx.cli semantic-audio --model models/yue2-8bit \
  --input recording.wav --output outputs/semantic-guided \
  --mode anchored --start-time 0.25 --noise-amount 0.1 --anchor-strength 2 \
  --reference reference.wav --reference-placement audio --steps 16
```

Install the separate weights once (about 2.85 GB downloaded):

```sh
.venv/bin/python -m yue2_mlx.cli install-realaudio
```

Or run without the web UI:

```sh
.venv/bin/python -m yue2_mlx.cli semantic-audio \
  --model models/yue2-8bit --input recording.wav \
  --output outputs/semantic-test --style 'piano, warm vocals' --steps 32
```

Use `--no-nar-adapter` for the stock-NAR comparison. The web tab exposes the
same switch. `--lyrics-file` is optional. `--tokenizer-model` selects a different
installed package directory in the CLI or web server. No training is needed.
No PyTorch, torchaudio, upstream checkout or remote Python code is used by
this inference path; original safetensors are loaded directly into MLX.

Implementation details:

- Pinned MERT-v2-FullSong: `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`.
  Mono mix at 24 kHz → checkpoint-normalized log-mel → ConvNeXt subsampler
  → first **21** Conformer blocks. Upstream `hidden_states[20]` has no input
  embedding entry. Features have width 1024 and approximately 25 frames/second.
- MERT uses the supplied recipe's **30-second chunks**, sequentially, not a
  batch of the entire song. Only CPU feature arrays survive MERT unloading.
  As upstream, subsecond final chunks are omitted before interpolation to the
  full track's frame count. This can stretch the retained features slightly;
  chunk boundaries can affect fidelity. Whole clips under one second are an
  extension: zero-pad for extraction, then retain only the original interval.
- Features are interpolated with half-pixel coordinates, rounded through FP16
  storage as upstream, and normalized per track with population standard deviation.
  The eight-layer, width-512 head uses overlapping 512-frame windows with
  central-region selection, predicting one of 32,768 codes at each frame.
- MERT and head arithmetic remain **FP32**. This is intentional, not bit-exact
  reproduction of the author's CUDA BF16 autocast. Near-tied token decisions
  can differ with precision. The VAE remains FP32; AR/NAR retain their configured
  checkpoint precision, typically 8-bit linears and BF16 activations.
- Pinned tokenizer/adapter: `841258e9d245257e069d6b0ec8c476effff9b98c`.
  NAR adapter rank-32 branches run in FP32 alongside the unchanged quantized
  base, with scale 1.0. Input/output projections are **full replacements**.
  This avoids requantizing merged weights; finite-precision results can differ
  from upstream's BF16 weight merge. AR and VAE weights are not modified.
- Semantic extraction collapses stereo to mono; synthesis creates stereo anew.
  This tab supports partial noising/source pull and optional reference memory,
  but not independent-channel processing. Existing Audio/Flow labs remain unchanged.

This is experimental **resynthesis**, not a trained restoration guarantee.
It can change lyrics, instrumentation, timing details and spatial placement.
Try short excerpts, compare adapter on/off with the same seed, and listen quietly.
The weights remain non-commercial CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.

## Local audio-only training

**Create music** and **Flow lab** also expose an **Acoustic checkpoint** selector:
Stock YuE2 (default), Mothersuperior, or a saved local checkpoint. Only the paired
NAR adapter/projections are applied; the AR composer, audio tokenizer and VAE
are not changed by this selector. Selection is saved in job history and restored
by Use settings. Flow selection applies to all methods, reference conditioning
and independent-channel runs; at start time zero NAR is bypassed as usual.
This reuse outside Semantic lab is experimental, not a claim of genre learning
or better generation quality. CLI `generate`, `synthesize`, and `flow-edit`
accept `--acoustic-checkpoint /path/to/checkpoint-package`; omit it for stock.

Open **Training** in the Studio sidebar (`/training`). Drop one or more songs,
remove any you do not want, name the run, choose its parameters and queue it.
Training shares the existing serial queue: it waits for current audio jobs and
does not run alongside them. Queued runs keep their own song-list snapshot;
removing a song from the collection does not change an already queued run or
delete its upload.

Your recordings need **no semantic labels or lyrics**. Preprocessing extracts
frozen MERT features and VAE latents, caches them on disk, and unloads those
models. Training adapts the tokenizer head plus the paired NAR LoRA and full
input/output projections. Gradients pass through the frozen AR to the head;
AR and NAR base weights remain unchanged. Layers are recomputed during backward
to reduce memory, and training windows are read from memory-mapped song arrays.

For retention, the pipeline automatically generates a separate small set of
clips with known tokens. This is a local adaptation recipe, **not an exact port
of the author's full training recipe**: it uses a small generated corpus,
hard-token cross entropy rather than neighbor-soft targets, and microbatches.
It cannot promise better fidelity or cleaner audio. With multiple songs the last
song is held out; with one song its last quarter is held out, which is not an
independent-song quality test. The displayed validation loss samples a fixed
window, not the entire held-out material.

Defaults are 500 updates, 256-frame windows (10.24 seconds), and a checkpoint
every 50 updates. For a first plumbing test, use 20 updates and save every 5;
that short run is not a quality benchmark. Initial extraction and generation of
the default four supplementary clips add setup time. Keep the Mac awake.

Each completed checkpoint pair appears automatically in Semantic lab's
**Tokenizer** selector beside Mothersuperior. Leave **Use selected checkpoint's
NAR adapter** enabled to use both trained components. An initial step-0 pair is
also saved as a baseline. Files live under
`outputs/web/<job-id>/output/checkpoints/step-XXXXXXXX/` with the default web root,
at approximately 300 MiB per pair. Only complete pairs are published. Run logs,
settings, validation metrics, and `best.json` remain in the job's output folder.

**Stop and save** finishes the current update and saves changed weights;
during preprocessing it stops after the current extraction/generation unit.
Force-quitting may lose updates since the last checkpoint. New checkpoints also
save optimizer moments/counters, the update position, random-generator state,
original settings/dataset and loss history. After stopping, click **Resume from
step N** on the run card to queue a continuation of the original schedule. It
uses the original songs/settings, not the current form, and runs only the
remaining updates. Completed runs have no remaining schedule to resume. Use
**Show older training runs** to find a past run. Preprocessing is cached and
models reload on resume; this is a saved pause, not a suspended process.

Each resumable checkpoint costs roughly another 600 MiB for optimizer state
when both components are trained. Older checkpoints without this state remain
weights-only. Selecting a checkpoint in **Start from** and submitting the form
still deliberately starts a **fresh optimizer and schedule**; use the run's
Resume button for continuation. Resume assumes unchanged base-model files and
the same implementation/runtime. Trained weights retain the upstream
non-commercial license restrictions.

Run cards show live flow, weighted tokenizer-retention and held-out flow loss
curves with independent vertical scales. Validation is sampled at checkpoint
intervals. These are noisy window losses, not perceptual audio-quality scores;
listen to held-out reconstructions as well. Continuations retain earlier loss
measurements up to the resumed checkpoint.

On the local 16 GiB M2, a real-weight 256-frame joint backward/update diagnostic
took 6.42 seconds and peaked at 5.70 GiB of MLX allocations. This used synthetic
features/latents and excludes preprocessing, validation, regularization, and
checkpoint I/O: it is a numerical/memory feasibility check, not a full-run
throughput or learning-quality measurement. MLX allocations exclude host memory.

## Experimental audio-to-audio flow lab

### Optional frozen reference audio

Flow lab has a **Reference audio** panel, independent of its three editing modes.
Drop one reference clip (at least 0.04 seconds), choose a placement, and set strength.
Both placements encode the reference, then run **one NAR prefill** to capture
per-layer reference K/V; the reference sees the text prompt and itself, never
the target. The cache stays frozen for the entire target solve. It is not
converted to semantic tokens and no predicted reference velocity is integrated.

- **Prefix:** attention layout / rotary positions `[AR, reference, target]`.
- **Acoustic:** `[AR, target, reference]`; reference positions follow the target.
- Both blocks retain their two boundary frames and their own local sinusoidal
  audio positions. There is no reference-specific trained marker or adapter.
- The cache is built at the target's starting time. Direct mode uses the clean
  encoded reference; other modes mix it with Gaussian noise once, using seed
  `target_seed XOR 0x524546`. It is **not** rebuilt as target time decreases.
- **Strength** adds `log(strength)` to reference-key attention logits. At 1,
  reference keys have ordinary attention weight; at 0, the entire reference
  path is bypassed, including positional shifts. This is not an output blend.
  More reference frames can attract more attention. Strength must be finite and nonnegative.
- Independent L/R mode uses the corresponding reference channel, duplicated to
  stereo, for each separate target run. Mono references duplicate to both sides.
- Start time zero bypasses all reference conditioning. Reference audio is saved
  for comparison; strength and placement are recorded in session history.

This is untrained reference-memory injection. It may copy melody/rhythm or
introduce artifacts rather than improve fidelity. Fixed-time cache mismatch and
the new positional layout are experimental, not guaranteed model capabilities.

```sh
yue2-mlx flow-edit --model models/yue2-8bit --input target.wav \
  --output outputs/reference-test --reference reference.wav \
  --reference-placement prefix --reference-strength 1 \
  --mode anchored --start-time 0.1 --steps 16
```

### Target processing

Both **Audio lab** and **Flow lab** offer **Channel processing → Independent L/R**.
Each channel is duplicated into a dual-mono input and processed sequentially;
the two output channels from each pass are averaged and restored to their
original left/right positions. Flow runs reuse the same seed and settings.
There is no per-channel normalization, padding, or time stretching. Model changes
can still affect balance/phase; this is not a guarantee of stereo preservation.
Joint stereo remains the default. Independent mode roughly doubles processing.
CLI: add `--independent-channels` to `reconstruct` or `flow-edit`.
Separate `left-*.npy` / `right-*.npy` latent files are saved instead of one
combined latent array, with detailed per-channel runs in `left/` and `right/`.

The **Flow lab** tab accepts a recording without an artificial duration cap. It compares the
resampled original, a VAE-only reconstruction, and a partial-flow result.
It does not extract semantic tokens from the recording or generate replacement
semantic tokens: the AR cache contains only the text prompt, and NAR processes
the encoded audio state. Blank style/lyrics still include the native instruction
and markers, so this is not strictly unconditional inference.

- **Noise + resume:** initialize `z(t) = (1-t) * encoded_audio + t * Gaussian`.
- **Direct refinement:** initialize with unchanged encoded audio at the chosen
  time, deliberately without matching its expected noise level.
- **Anchored noise + resume:** same initialization as noise mode, then pull
  toward `(1-t)*source + t*original_noise` after each solver step, at the new time.
  Source pull defaults to 2 (try 1–3); zero is exactly ordinary noise mode.
  The blend per step is `1-exp(-strength*dt)`, so strength is a rate per unit
  flow time, not a fixed percentage per step. Finite-step solver results can
  still vary with step count. Strong pulls preserve source defects too.
- **Start time:** integrate from this value to zero. Try 0.05–0.20; zero bypasses
  the transformers. In noise mode, one removes all source information.
- **Remaining steps:** midpoint steps over that interval (two NAR evaluations
  per step), not a count of forward-noising steps. Noise is drawn once.

This is an experimental straight-path initialization, not a verified reproduction
of the checkpoint's training corruption distribution or a trained denoiser.
Original recording noise does not automatically correspond to flow time.
Neither mode guarantees cleanup or preservation; listen quietly at first.
There is no ODE inversion or fresh per-step noise. Only anchored mode applies
repeated source pulls; it reuses the original noise sample and never forces
the final state to equal the source exactly.
The idea is related to [SDEdit](https://arxiv.org/abs/2108.01073), but this
implementation uses the existing deterministic midpoint flow solver.

```sh
yue2-mlx flow-edit --model models/yue2-8bit --input recording.wav \
  --output outputs/flow-experiment --mode mix --start-time 0.15 --steps 16 \
  --style 'piano, natural room sound' --seed 831001
```

For source anchoring use `--mode anchored --anchor-strength 2` instead of
`--mode mix`. The report records the pull strength and reference rule.

Outputs include `source-latents.npy`, `initial-latents.npy`, final `latents.npy`,
`reconstruction.wav`, `flow-edit.wav`, `input-resampled.wav`, and a settings/report
JSON. One weight stage is resident at a time. Only the released shift=1 schedule
is supported for this experiment. Longer clips are rejected, not silently cut.

## Install

Apple silicon and a current macOS/MLX installation are required for Metal execution.

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

## Local web studio

Double-click **Launch YuE2 Studio.command** in Finder. It opens Terminal, starts
the local server, and opens your browser at **http://127.0.0.1:7860**. Keep that
Terminal window open; Ctrl+C stops the server and any running inference job.

Alternatively:

```sh
source .venv/bin/activate
python -m yue2_mlx.web --open
```

The studio provides song generation, drag-and-drop lyrics/ABC scores, audio
uploads for reconstruction, original/reconstruction playback, downloads, live
progress, cancellation, and persistent session history. It runs one job at a time
in a separate process to release Metal memory when a job ends or is stopped.
The audio lab is a reconstruction experiment, not a trained noise-removal tool.

You can submit jobs while another is running: the button changes to **Add to
queue**. Generation, Audio lab, and Flow lab share one first-in, first-out queue.
Each submission saves its own inputs, reference, and settings; later form changes
do not affect it. History shows waiting positions and **Remove from queue**.
Stopping a running job or a failed job advances to the next one automatically.
Waiting jobs survive server restarts and resume after the server starts; an
interrupted in-progress job is not automatically retried. Closing the browser
does not stop processing. Keep the server Terminal open and the Mac awake for
overnight runs.

Each new session retains its settings and launch command in `job.json`.
History cards show a compact settings summary; expand it for full prompts,
parameters, and **Use settings**. Reuse restores the form and available input
audio without starting a job. Older sessions recover whatever their artifacts
recorded and are marked as partial/recovered. Reuse uses the current server's
model/runtime; it is not a guarantee of bit-identical output across versions.

Defaults use `models/yue2-8bit` and `models/yue2-vae`; override with `--model`
and `--vae`. Sessions and uploaded audio stay under `outputs/web/`. The server
binds to localhost only and uses no hosted services. Uploads must be mono/stereo
and at least one latent frame (0.04 seconds); there is no app-imposed upload size
or duration cap. Supported codecs depend on libsndfile.

Duration, prompt length, step count, and conditioning strengths have no artificial
upper caps. The model's 24,576-position context checks still apply, including the
combined prompt, target, and reference in Flow lab. Available memory and disk space
remain practical limits; long references also make attention more expensive.
The configurable MLX allocator budget remains 10 GiB by default (`--memory-gib`
when launching the server or CLI). Flow time stays in [0, 1] and seeds are uint32.

## Convert weights

```sh
yue2-mlx convert --output models/yue2-8bit --bits 8 --nar-bits 8
```

This downloads `m-a-p/YuE2-3B` and `m-a-p/YuE2-Vae`. Use `--model` / `--vae`
for local original checkpoint directories; `--model-revision` / `--vae-revision`
pin Hugging Face revisions. Sources may have one safetensors file or an index
of shards. Conversion reads individual tensors and writes small stage shards;
it does not construct the source transformer, and needs no PyTorch.

Use `--bits 0 --nar-bits 0` for BF16 transformer weights, or `--bits 4 --nar-bits 8`
for experimental mixed precision. Quantization changes predictions; 4-bit acoustic
generation especially needs listening evaluation. Embeddings, norms, timestep
embedding and latent projections remain BF16. The VAE remains FP32.

For the audio experiment alone, skip the large transformer entirely:

```sh
yue2-mlx convert --vae-only --output models/yue2-vae
yue2-mlx reconstruct --model models/yue2-vae --input recording.wav --output outputs/roundtrip
```

`roundtrip` retains a resampled stereo input, encoded latents, reconstruction, and
memory report. Mono is duplicated to stereo; other channel counts are rejected.
The encoder uses the deterministic posterior mean. Natural decoder output length
is preserved, including its small end-length difference; no tail is silently padded.

## Generate a song

Put lyrics with section tags in a UTF-8 text file:

```sh
yue2-mlx generate --model models/yue2-8bit \
  --style "English, piano pop, warm lead vocal" --lyrics-file examples/lyrics.txt \
  --output outputs/song
```

`--cot full` (default) generates melody and chord ABC first. `--cot melody` gives
melody planning and `--cot off` skips score planning. Use `--abc-file score.abc`
with full/melody for a supplied composition. To obtain semantic conditioning
from a recording instead, use the optional `semantic-audio` command or Semantic lab
described above. This uses Mothersuperior's learned tokenizer, not an official
release of YuE2's original audio tokenizer.

Useful controls: `--seed`, `--cfg-scale`, `--steps` (default 32),
`--max-semantic-tokens` (default 9000, approximately 360 seconds at 25 Hz),
and `--min-semantic-tokens` (default 200). For a short smoke test set both token
limits explicitly, for example 250 maximum and 50 minimum. Reaching a token cap
is recorded as truncation, not reported as natural completion. Fewer flow steps
change the synthesis and may reduce quality.

The output directory retains plan/semantic JSON, score ABC, `latents.npy`,
`audio.flac`, and a result/memory report. Outputs must be new paths; failures
retain intermediate files. MLX seeds are repeatable within the same runtime,
but do not reproduce PyTorch RNG samples or guarantee identical results across
hardware/versions. No top-level code loads or depends on the ignored `YuE/` folder.

## Memory lifecycle

| Stage | Resident weights | What survives the stage |
|---|---|---|
| Plan / semantic generation | AR only | Token IDs and score |
| Acoustic conditioning | AR, without output head | Evaluated per-layer keys/values |
| Acoustic synthesis | NAR only | Acoustic latents |
| Audio encoding | VAE encoder only | Acoustic latents |
| Audio decoding | VAE decoder only | Streamed waveform tiles |

Weights are genuinely released, not moved to CPU (which does not save unified
RAM). Acoustic chunks reload the AR and NAR stages sequentially to avoid holding
both networks or all conditioning caches. This trades loading time for memory.
The small shared final norm is stored in both stages. Position embeddings are
computed as needed rather than loading the full fixed table.

Defaults target a conservative 16 GB workflow: a **10 GiB MLX allocator limit**,
256 MiB allocator cache, 256-query attention tiles, and 256-frame VAE cores.
These are controls, not a demonstrated guarantee of 16 GB full-song operation.
An actual eight-second 8-bit generation on a 16 GB M2 peaked at **2.54 GiB of
MLX allocations**; the full command, timings and limits are in [VALIDATION.md](VALIDATION.md).
NumPy, Python, macOS and file-backed pages also consume RAM. `--memory-gib`,
`--cache-mib`, `--query-chunk`, and `--vae-core` are configurable. Attention tiling
retains the original visible keys; VAE tiling computes required context from its
receptive field, crops exact cores, and does not use crossfades.

Memory events report MLX peak allocations, not whole-process RSS. The allocator
limits are process-wide; use one pipeline/request at a time. Long input recordings
are currently read into host NumPy memory before tiled encoding.

## Individual stages and Python

CLI commands `plan`, `semantic`, `synthesize`, and `decode` allow resuming from
saved artifacts. `--input` points to `plan.json`, `semantic.json`, or `latents.npy`
as appropriate. Use `yue2-mlx COMMAND --help` for arguments.

```python
from yue2_mlx import Pipeline

pipe = Pipeline("models/yue2-8bit")
plan = pipe.plan(style="Jazz, piano trio", lyrics="[Verse]\nA quiet evening")
semantic = pipe.generate_semantic(plan)
latents = pipe.synthesize(semantic)
pipe.decode(latents, "outputs/example.wav")
```

For array experiments, `pipe.encode(stereo_numpy_audio, 48000)` returns latent
frames. The loaded `AudioNetwork` also exposes differentiable encoder/decoder
calls; the high-level NumPy/file pipeline is inference-only. General latent editing,
flow inversion, and partial-noise audio regeneration are not implemented yet.

## Validation

The consolidated suite compares shared random weights/inputs with an independent
upstream checkout: FP32/BF16 AR logits, incremental and chunked caches, NAR
conditioning and velocities, midpoint trajectories, VAE encoding/decoding and
tile seams, sampling masks, sharded conversion, quantization, and stage unloading.

```sh
uv pip install --python .venv/bin/python -e '.[test]'
pytest tests --reference /absolute/path/to/upstream/YuE
```

Without `--reference`, upstream comparisons are explicitly skipped. The upstream
source is never bundled. Random-weight parity validates implementation, not trained
model quality or full-song memory. See `VALIDATION.md` for actual run results.

## Attribution

Architecture and native protocol: the YuE2 authors, original reference revision
`88da114a67df892af0329472073b96a5ef700b93`. Source adaptations retain Apache-2.0
licensing; see `LICENSE` and `THIRD_PARTY_NOTICES.md`. Oobleck and SnakeBeta retain
their MIT notices. Original model weights remain separately licensed CC BY-NC 4.0.
See the [original model card](https://huggingface.co/m-a-p/YuE2-3B).
