# Third-party notices

This standalone MLX implementation adapts the YuE2 inference equations,
checkpoint topology and native prompting/sampling protocol from
https://github.com/multimodal-art-projection/YuE at
88da114a67df892af0329472073b96a5ef700b93.
Copyright (c) 2026 the YuE2 authors. Licensed under Apache License 2.0;
the complete text is in LICENSE. The upstream checkout is not distributed.

Changes here include independent MLX modules, separate AR/NAR checkpoint
storage/loading, encoder and decoder weight conversion, channels-last
convolutions, tiled encoding/decoding, staged memory management, and a CLI.
No code from the ahmadw community MLX port is included.

The Oobleck topology and SnakeBeta equations follow the upstream implementation
derived from stable-audio-tools commit a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45:

- Oobleck / stable-audio-tools: Copyright (c) 2023 Stability AI, MIT.
  Full notice: licenses/stable-audio-tools-MIT.txt.
- SnakeBeta / BigVGAN: Copyright (c) 2022 NVIDIA CORPORATION, MIT.
  Full notice: licenses/SnakeBeta-NVIDIA-MIT.txt.

Weights are not bundled. Original, converted, and locally adapted YuE2 weights
are subject to the original CC BY-NC 4.0 terms, separately from implementation
code.

## Optional real-audio semantic tokenizer

The independent MLX implementation in `realaudio.py` follows the published
MERT2 topology and Mothersuperior v4 inference recipe. Attribution:

- MERT2 / m-a-p, MERT-v2-FullSong:
  https://huggingface.co/m-a-p/MERT-v2-FullSong
  revision `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`.
- Mothersuperior, yue2-mothersuperior-realaudio-tokenizer-v4:
  https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4
  revision `841258e9d245257e069d6b0ec8c476effff9b98c`.
- YuE2 / m-a-p, the base model and semantic vocabulary.

The optional downloader pins these revisions and retains MERT's LICENSE and
THIRD_PARTY_NOTICES.md and Mothersuperior's model card in the local model cache.
MERT, tokenizer-head and NAR-adapter weights are CC BY-NC 4.0 (non-commercial),
not Apache-licensed by this project. No upstream training scripts are bundled
or executed by the application. PyTorch/torchaudio are used only in optional
development parity tests and retain their own licenses.

The native MLX training implementation in `training.py` follows Mothersuperior's
published joint-adaptation approach (straight-through tokenizer predictions,
frozen AR conditioning, NAR adapters and generated-data retention). Its small
automatically generated retention set, hard-token loss and microbatching are
documented local modifications, not a claim of reproducing the original recipe.
Locally adapted checkpoints remain subject to the source weights' license;
training does not remove their non-commercial restrictions.
