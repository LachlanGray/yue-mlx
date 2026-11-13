# Validation record

## Combined semantic flow and bounded playback history — September 15, 2026

- Semantic lab now supports fresh, mix, direct and anchored initializations,
  independent optional noise amount, start time, source pull and either frozen
  reference-cache placement. Source/noise arrays are drawn once before acoustic
  chunking. Reference memory is rebuilt once per chunk, not per solver step.
- Six new pipeline cases verify initialization amounts, automatic time coupling,
  direct/no-noise mode, zero-time bypass, source anchoring, semantic IDs in the
  actual AR input, and reference forwarding. MERT/VAE frame counts are aligned
  by cropping their common interval.
- Full suite with `--reference YuE`: **64 passed, 1 skipped**, 94.27 seconds.
  The optional pretrained VAE test remains skipped without source flags.
  An additional final HTTP combined-request test was then added; the targeted
  semantic-flow + web suite passed **18 tests**, including that request and
  rejection of invalid custom noise amounts.
- Chrome tests pass for combined controls, reference replacement, history reuse,
  queue cancellation with active-first ordering, and history pagination. Main
  workspace mounts active/queued jobs plus three completed results; `/history`
  mounts six completed results per page. Leaving a page pauses players, removes
  their sources and discards their DOM/cards. No historical audio is deleted.
- Actual 8-second combined smoke: anchored, t=0.25, noise amount=0.1, strength=2,
  v4 NAR adapter, acoustic-placement reference, 8 midpoint steps. Result has
  383,936 finite stereo samples at 48 kHz; peak MLX allocation 2.1525 GiB.
  Artifacts: `outputs/semantic-combined-smoke/`. Quality is for user evaluation;
  this verifies execution, not improved restoration.

## Real-audio semantic tokenizer — September 15, 2026

Implemented native MLX MERT-v2-FullSong (through Conformer block index 20),
Mothersuperior v4 tokenizer head, and optional real-audio NAR adapter branches
plus full input/output projection replacements. No torch dependency at inference.
Pinned revisions and preprocessing/precision differences are in README.md.

- Full regression suite: **58 passed, 1 skipped** in 98.01 seconds using
  `.venv/bin/python -m pytest -c pyproject.toml tests --reference YuE -q`.
  The skipped test is the optional pretrained VAE comparison; its source flags
  were not provided. Existing AR/NAR, VAE topology, conversion and pipeline
  tests did run. Warnings are upstream weight_norm and nested-tensor notices.
- New tests include transformer-head FP32 parity, ConvNeXt/GRN parity,
  overlapping-window coverage across edge lengths, LoRA arithmetic over an
  8-bit base, and semantic-job queue/settings/validation.
- Full **pretrained** CPU PyTorch vs MLX FP32 check using the pinned MERT source:
  mel max absolute error **7.15e-7**; MERT output features **7.72e-5**;
  tokenizer-head logits **1.91e-5**; **100% top-1 agreement** on the test window.
  These are numerical checks on a one-second synthetic input, not an audio
  quality or general token-accuracy benchmark. See `tests/parity_realaudio.py`.
  A separate ignored `models/realaudio-parity-env` contains matched torch and
  torchaudio 2.10.0 because the app venv's torch 2.14 has no matching torchaudio
  release. The app environment was not downgraded. Run with:

  ```sh
  PYTHONPATH=src models/realaudio-parity-env/bin/python tests/parity_realaudio.py
  ```

- Headless Chrome mocked-API tests pass for the new tab: drag/drop and replace,
  adapter on/off, submit while busy, immutable queued settings, history reuse,
  and missing-weight readiness. Existing queue, reference and history browser
  regressions also pass (`tests/browser_realaudio.py`, `browser_queue.py`,
  `browser_anchor.py`, `browser_history.py`).
- **Actual browser → HTTP → subprocess → MLX** test queued two 8-second,
  32-step jobs, adapter on then stock NAR, in an isolated test history. Both
  completed with finite 48 kHz stereo output, 383,936 samples, matching this
  source excerpt. All six original/baseline/result players appeared. Recorded
  model-stage times totalled about **45.1 s adapted / 41.8 s stock**, both with
  **2.1525 GiB peak MLX allocation**. This is not total process/OS memory or a
  long-song speed benchmark. Results/screenshots: `outputs/semantic-ui-check/`.
  See `tests/browser_realaudio_live.py` (requires a fresh output directory).
- An earlier 8-step end-to-end smoke is retained in
  `outputs/semantic-audio-v4-smoke/`. Tests used our existing generated excerpt;
  no subjective real-recording cleanup improvement is claimed.

The normal studio server/queue was not restarted or modified by these tests.
Tokenizer weights and package manifest are installed in `models/` ready for use.
The offline guide now includes `context/realaudio.d2` and its rendered SVG.

## Web studio — September 13, 2026

Four local HTTP integration tests pass: uploads and invalid files, request
validation/origin checks, single-job enforcement/cancellation/history, and byte
ranges for audio seeking. Chrome browser checks exercised dropped lyrics,
an uploaded recording, actual pretrained VAE reconstruction, actual music
generation, playback, and a mobile viewport without horizontal overflow or
JavaScript errors. Browser checks are optional scripts in `tests/browser_smoke.py`
and `tests/browser_generate.py` (Playwright required only for development).
Screenshots and resulting audio are retained locally under `outputs/web-check/`
and `outputs/web/`. The executable Finder launcher passes shell syntax checks;
the wheel includes all frontend assets.

## Inference engine — September 12, 2026

Run on an Apple M2 with **16 GiB unified memory**.
Python 3.13.12, MLX 0.32.2, PyTorch 2.14.0, Transformers 4.57.6.
Upstream reference: YuE2 commit `88da114a67df892af0329472073b96a5ef700b93`.

## Consolidated suite

**18 tests passed** (75.43 seconds). Tests cover:

- RMSNorm, RoPE, grouped attention and absolute causal offsets.
- FP32/BF16 AR logits with full, incremental and chunked prefill.
- FP32/BF16 NAR conditioning keys/values, velocities and midpoint trajectories.
- VAE encoder, decoder, natural lengths, exact tiled cores and optional final tanh.
- FP32 and BF16 checkpoint reading, sharded indexes, HF snapshot symlinks,
  separate stage loading and 4/8-bit conversion.
- Native sampling constraints and a full reduced-size generation using the
  real-sized vocabulary, CFG, flow matching and waveform output.
- Actual release of materialized model weights between stages.
- Pretrained VAE parity described below.

The warnings come from deprecated weight normalization in the upstream PyTorch
reference. They do not indicate failed checks. The test suite can be rerun with:

```sh
pytest tests --reference /path/to/upstream/YuE \
  --vae-source /path/to/original/YuE2-Vae \
  --converted-vae models/yue2-vae
```

Omit the last two arguments to skip the pretrained test. Omit `--reference` to
skip upstream comparisons. Check the skip count before interpreting a test run.

## Pretrained VAE

Original VAE revision: `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`.
The original encoder/decoder weights were converted and tested in FP32 against
the upstream PyTorch implementation. Input was one second of harmonic synthetic
audio, with added noise in one stereo channel. Encoding and decoding used
multiple tiles to exercise boundaries.

| Comparison | Relative L2 error |
|---|---:|
| Encoder latents | 1.278e-6 |
| Decoder waveform, identical latents | 1.030e-6 |
| Complete round trip | 1.169e-6 |

The 48,000-sample input encoded to 25 frames; the decoder produced its natural
47,936-sample output. No padding was used to conceal the length difference.

These measurements establish close numerical agreement, **not denoising quality**.
The experiment does not establish improvement on real noisy recordings.

The actual CLI round trip is retained in `outputs/vae-smoke/` locally (ignored
by git). On that one-second input, the encoder peaked at **0.413 GiB** and the
decoder at **0.453 GiB** of MLX allocations. Both stages returned to effectively
zero active MLX memory after unloading. Total stage times, including loading,
were approximately 0.24 and 0.30 seconds. These short-input figures should not be
extrapolated to long recordings or interpreted as whole-process RAM usage.

## Frozen audio-reference memory (September 14, 2026)

- 24 tests passed across reference attention, flow editing, independent-channel
  routing, and web validation/history. Includes frozen K/V immutability,
  target-independent prefill, zero-strength exact bypass, both positional
  layouts, attention-logit strength weighting, and combined source anchoring.
- Three targeted FP32/BF16 no-reference PyTorch parity tests passed after the
  attention/solver extension.
- Browser tests cover reference drag/drop and replacement, removal, settings
  payloads, and restoring both target/reference uploads from session history.
- Real 8-bit checkpoint smoke runs: prefix placement with noise/resume, and
  acoustic placement with anchoring plus independent L/R. Both produced finite
  47,936-sample stereo 48 kHz outputs on the one-second synthetic test recording.
  Highest recorded stage allocation was approximately 2.105 GiB in each run.
  Local outputs: `outputs/reference-prefix-smoke` and
  `outputs/reference-audio-independent-smoke-v2`.

These short tests do not establish cleanup quality, successful style transfer,
or maximum-context memory usage. Reference K/V are acoustic-derived, not
extracted semantic tokens. They stay frozen at the chosen initial flow time.

## Packaging

The wheel builds successfully and contains only the standalone `yue2_mlx`
package, metadata and license notices. Neither the reference checkout nor
model weights are bundled. Bytecode compilation and CLI argument help pass.

## Pretrained generator smoke test

Original transformer revision: `29b3558dd46954a0cd9021dc76d5c91864a0f1c7`.
The actual YuE2-3B checkpoint converted successfully, including all expected
parameter names and shapes. Converted AR weights occupy 2.47 GiB and NAR
weights 1.40 GiB, loaded separately. Both paths use 8-bit affine linear weights.

```sh
yue2-mlx generate --model models/yue2-8bit \
  --style 'English, piano pop, warm lead vocal' \
  --lyrics-file examples/lyrics.txt --cot off \
  --max-semantic-tokens 200 --min-semantic-tokens 200 \
  --output outputs/generation-smoke
```

This completed with the default 32 midpoint steps, off-mode CFG 1.01, and the
10 GiB allocator limit on the 16 GiB M2. It produced 383,936 stereo samples at
48 kHz (7.999 seconds), with finite values, RMS 0.1185 and peak amplitude 0.7459.
The sample is in `outputs/generation-smoke/audio.flac` locally, ignored by git.
It is deliberately token-capped and correctly reports truncation; it is not a
naturally completed song or a subjective quality evaluation.

| Stage | Peak MLX allocations | Time including stage loading |
|---|---:|---:|
| Semantic generation with CFG | 2.543 GiB | 26.01 s |
| AR conditioning, without vocabulary head | 2.144 GiB | 1.06 s |
| NAR synthesis, 32 midpoint steps | 1.457 GiB | 31.51 s |
| VAE decoding | 1.620 GiB | 1.09 s |

The largest measured allocation peak was **2.543 GiB**. Active MLX memory
returned to effectively zero after semantic generation and decoding. Only the
0.028 GiB conditioning cache survived the AR-conditioning/NAR stage boundaries;
the pipeline released it after that acoustic chunk.

Native `cot=full` score planning was also checked separately with the real
checkpoint and example lyrics. It emitted **413 ABC tokens and a natural end
token**, with `truncated=false`. The score contains the native vocal/instrumental
voices and chord annotations; the saved plan is `outputs/plan-smoke.json`.

## Limits

Random-weight transformer comparisons establish architecture parity, not trained
song quality. Quantized checks on reduced models are not a listening evaluation
of the full quantized generator. MLX allocator peaks exclude host allocations
and are not whole-process RSS. Full-song and maximum-context memory have not
been established by the tests above.
# Cap-removal regression (September 14, 2026)

Removed artificial duration, upload-size, prompt-length, step-count, and strength
ceilings. The actual model context, valid numeric domains, and configurable MLX
allocator budget remain enforced.

- Reference, flow-edit, independent-channel, and web suites: **26 passed**.
- Web regression accepts a 601-second upload as target and reference, a prompt
  above the former request-size limit, 500-second generation settings, 96 steps,
  reference strength 6, and source pull 25. Job execution is mocked in this test;
  this validates request handling, not long-clip memory use or audio quality.
- Headless Chrome: reference drag/drop, replacement/removal, both placements,
  values above former caps, independent channels, history reuse, and repeated
  audio-lab drops passed. Browser API responses are mocked.
# Persistent job queue (September 14, 2026)

Queue, web, reference, flow-edit, and independent-channel suites: **29 passed**.
Controlled real subprocess tests cover FIFO dispatch with no overlapping workers,
successful completion, nonzero exits, launch failures, running/queued cancellation,
shutdown without dispatch, and explicit resumption of persisted waiting jobs.
Constructing another Studio instance does not start work before server binding.
Headless Chrome checks (mocked API) passed for submission while busy, changed audio
and parameters, queue positions and removal, plus existing reference/history UI.
These are orchestration tests, not additional full-size model quality tests.

## Local MLX training (September 15, 2026)

Targeted training, real-audio, semantic-flow, web and queue tests: **40 passed**.
Tests cover straight-through hard token embeddings, finite nonzero gradients
through frozen AR and quantized base layers into the head/LoRA/projections,
deterministic training dropout, and a two-update tiny-model optimizer run.
Checkpoint publication, held-out evaluation, graceful stop/save, immutable
queued dataset snapshots and checkpoint discovery are exercised. Tiny-model
training uses prepared synthetic arrays, not a full real-song training run.

Headless Chrome checks passed for multi-song drop/upload, removal, parameter
submission while busy, checkpoint choices appearing automatically, and existing
Semantic lab, pagination, history reuse and queue interactions. Browser API
responses are mocked; these checks do not measure training or audio quality.

`tests/preflight_training.py` additionally completed a real-weight 256-frame
joint backward and Adam update on the 16 GiB M2, using the installed 8-bit AR/NAR
and original v4 tokenizer/adapter. All gradients and updated weights were finite:
loss 0.484502, gradient norm 1.674493, **77,946,432 trainable values**, **6.419 s**,
**5.697 GiB peak MLX allocations**. Inputs were synthetic features/latents. No
updated weights were saved or registered. This is not a learning-quality test
or full training throughput measurement: it excludes preprocessing, the separate
retention loss, validation and checkpoint writes, and excludes host allocations.

### Resumable training and loss charts

Training/web/queue regression suite: **23 passed**. The interrupted small-model
test saves at update 1, reloads weights, optimizer state and RNG, then continues
to update 2. Every saved weight and optimizer tensor is exactly equal to the
uninterrupted two-update run. This verifies the continuation logic on the local
runtime, not cross-version/cross-hardware bitwise reproducibility. API tests
cover original dataset/settings preservation, duplicate active continuation
rejection, incomplete checkpoints, and partially written metric lines.
Headless Chrome also passed rendering all three loss plots and submitting the
run-specific Resume action, alongside the existing upload/checkpoint-selector
checks (mocked API).
