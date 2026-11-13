"""Conversion, generation, resumable stages, and VAE round-trip experiments."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description="YuE2 in MLX: one model stage resident at a time")
    sub = parser.add_subparsers(dest="command", required=True)
    cv = sub.add_parser("convert", help="Convert original HF weights into separately loadable stages")
    cv.add_argument("--model", default="m-a-p/YuE2-3B")
    cv.add_argument("--vae", default="m-a-p/YuE2-Vae")
    cv.add_argument("--output", required=True)
    cv.add_argument("--bits", type=int, choices=[0, 4, 8], default=8, help="AR linear precision; 0 = BF16")
    cv.add_argument("--nar-bits", type=int, choices=[0, 4, 8], default=8)
    cv.add_argument("--group-size", type=int, choices=[32, 64, 128], default=64)
    cv.add_argument("--vae-only", action="store_true", help="Download/convert only the audio encoder and decoder")
    cv.add_argument("--model-revision")
    cv.add_argument("--vae-revision")
    setup = sub.add_parser("install-realaudio", help="Download pinned MERT and v4 tokenizer/adapter data")
    setup.add_argument("--output", default="models/realaudio-v4")
    imp = sub.add_parser("import-training-bundle", help="Convert upstream tokenizer/NAR/AR .pt outputs")
    imp.add_argument("--source", required=True)
    imp.add_argument("--output", required=True)
    imp.add_argument("--base-tokenizer", default="models/realaudio-v4")
    arimp = sub.add_parser("import-ar-checkpoint", help="Convert one upstream AR LoRA .pt checkpoint")
    arimp.add_argument("--source", required=True)
    arimp.add_argument("--output", required=True)
    arimp.add_argument("--name")
    for command in ("generate", "plan", "semantic", "synthesize", "decode", "reconstruct", "flow-edit", "semantic-audio"):
        p = sub.add_parser(command)
        p.add_argument("--model", required=True, help="Converted checkpoint directory")
        p.add_argument("--output", required=True, help="New output directory (or file for individual stages)")
        p.add_argument("--memory-gib", type=float, default=10)
        p.add_argument("--cache-mib", type=int, default=256)
        p.add_argument("--query-chunk", type=int, default=256)
        p.add_argument("--vae-core", type=int, default=256)
        p.add_argument("--quiet", action="store_true")
        if command in ("generate", "synthesize", "flow-edit", "semantic-audio"):
            p.add_argument("--acoustic-checkpoint", help="Tokenizer package whose paired NAR adapter to use; omit for stock")
            p.add_argument("--acoustic-scale", type=float, default=1.)
        if command in ("generate", "plan", "semantic", "synthesize", "flow-edit", "semantic-audio"):
            p.add_argument("--ar-checkpoint", help="Package containing an AR LoRA")
            p.add_argument("--ar-scale", type=float, default=1.)
        if command in ("generate", "plan"):
            p.add_argument("--style", required=True)
            p.add_argument("--lyrics-file", required=True)
            p.add_argument("--cot", choices=["full", "melody", "off"], default="full")
            p.add_argument("--abc-file")
            p.add_argument("--seed", type=int, default=831001)
        if command in ("generate", "semantic"):
            p.add_argument("--max-semantic-tokens", type=int, default=9000)
            p.add_argument("--min-semantic-tokens", type=int, default=200)
            p.add_argument("--cfg-scale", type=float)
        if command == "flow-edit":
            p.add_argument("--style", default="")
            p.add_argument("--lyrics-file")
            p.add_argument("--start-time", type=float, default=.15)
            p.add_argument("--mode", choices=["mix", "direct", "anchored"], default="mix")
            p.add_argument("--anchor-strength", type=float, default=2.)
            p.add_argument("--reference", help="Optional reference audio (shares the model context)")
            p.add_argument("--reference-placement", choices=["prefix", "audio"], default="prefix")
            p.add_argument("--reference-strength", type=float, default=1.)
            p.add_argument("--seed", type=int, default=831001)
        if command == "semantic-audio":
            p.add_argument("--mode", choices=["fresh", "mix", "direct", "anchored"], default="fresh")
            p.add_argument("--start-time", type=float, default=1.)
            p.add_argument("--noise-amount", type=float, help="Override source/noise mix independently of start time (0–1)")
            p.add_argument("--anchor-strength", type=float, default=2.)
            p.add_argument("--reference")
            p.add_argument("--reference-placement", choices=["prefix", "audio"], default="prefix")
            p.add_argument("--reference-strength", type=float, default=1.)
            p.add_argument("--tokenizer-model", default="models/realaudio-v4")
            p.add_argument("--no-nar-adapter", action="store_true")
            p.add_argument("--style", default="")
            p.add_argument("--lyrics-file")
            p.add_argument("--seed", type=int, default=831001)
        if command in ("reconstruct", "flow-edit"):
            p.add_argument("--independent-channels", action="store_true")
        if command in ("generate", "synthesize", "flow-edit", "semantic-audio"):
            p.add_argument("--steps", type=int, default=32)
        if command in ("semantic", "synthesize", "decode", "reconstruct", "flow-edit", "semantic-audio"):
            p.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "install-realaudio":
            from .realaudio import install
            install(args.output)
            print(args.output)
            return
        if args.command == "import-training-bundle":
            from .adapters import import_training_bundle
            import_training_bundle(args.source, args.output, args.base_tokenizer)
            print(args.output)
            return
        if args.command == "import-ar-checkpoint":
            from .adapters import import_ar_checkpoint
            import_ar_checkpoint(args.source, args.output, args.name)
            print(args.output)
            return
        if args.command == "convert":
            from .weights import convert
            convert(args.model, args.vae, args.output, bits=args.bits, nar_bits=args.nar_bits,
                    group_size=args.group_size, vae_only=args.vae_only,
                    model_revision=args.model_revision, vae_revision=args.vae_revision)
            return
        from .pipeline import Pipeline
        from .protocol import Sampling
        import numpy as np
        last = [0.]
        def progress(stage, done, total):
            now = time.monotonic()
            if not total or done == total or now - last[0] >= 2:
                print(f"{stage}" + (f": {done}/{total}" if total else ""), file=sys.stderr, flush=True)
                last[0] = now
        pipe = Pipeline(args.model, memory_gib=args.memory_gib, cache_mib=args.cache_mib,
                        query_chunk=args.query_chunk, vae_core=args.vae_core, progress=None if args.quiet else progress)
        output = Path(args.output)
        if getattr(args,'acoustic_checkpoint',None):
            pipe.realaudio_adapter = args.acoustic_checkpoint
        if hasattr(args, 'acoustic_scale'):
            pipe.realaudio_adapter_scale = args.acoustic_scale
        if getattr(args, 'ar_checkpoint', None):
            pipe.ar_adapter = args.ar_checkpoint
            pipe.ar_adapter_scale = args.ar_scale
        if output.exists():
            raise FileExistsError(output)
        kwargs = {}
        if args.command in ("generate", "plan"):
            kwargs = dict(style=args.style, lyrics=Path(args.lyrics_file).read_text(), cot=args.cot,
                          abc=Path(args.abc_file).read_text() if args.abc_file else None, seed=args.seed)
        if args.command in ("generate", "semantic"):
            kwargs.update(sampling=replace(Sampling(), max_tokens=args.max_semantic_tokens,
                                          min_tokens=args.min_semantic_tokens), cfg_scale=args.cfg_scale)
        if args.command == "generate":
            pipe.generate(output, steps=args.steps, **kwargs)
        elif args.command == "reconstruct":
            pipe.reconstruct(args.input, output, independent_channels=args.independent_channels)
        elif args.command == "semantic-audio":
            pipe.semantic_audio(args.input, output, tokenizer_model=args.tokenizer_model,
                                mode=args.mode, start_time=args.start_time, noise_amount=args.noise_amount,
                                anchor_strength=args.anchor_strength, reference_path=args.reference,
                                reference_placement=args.reference_placement, reference_strength=args.reference_strength,
                                nar_adapter=not args.no_nar_adapter, style=args.style,
                                lyrics=Path(args.lyrics_file).read_text() if args.lyrics_file else "",
                                steps=args.steps, seed=args.seed)
        elif args.command == "flow-edit":
            pipe.flow_edit(args.input, output, style=args.style,
                           lyrics=Path(args.lyrics_file).read_text() if args.lyrics_file else "",
                           start_time=args.start_time, mode=args.mode, steps=args.steps, seed=args.seed,
                           anchor_strength=args.anchor_strength, independent_channels=args.independent_channels,
                           reference_path=args.reference, reference_placement=args.reference_placement,
                           reference_strength=args.reference_strength)
        elif args.command == "decode":
            pipe.decode(np.load(args.input, allow_pickle=False), output)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            if args.command == "plan":
                result = pipe.plan(**kwargs)
            elif args.command == "semantic":
                result = pipe.generate_semantic(json.loads(Path(args.input).read_text()), **kwargs)
            else:
                result = pipe.synthesize(json.loads(Path(args.input).read_text()), steps=args.steps)
            if isinstance(result, np.ndarray):
                with output.open("xb") as f:
                    np.save(f, result)
            else:
                with output.open("x") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        print(output)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
