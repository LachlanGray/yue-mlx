"""Native YuE2 prompts and vocabulary boundaries (YuE2 authors, Apache-2.0)."""
import base64
from dataclasses import dataclass
from pathlib import Path
import math
import unicodedata

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
CONTEXT = 24576
INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
    "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
}


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_p: float = .95
    top_k: int = 100
    repetition_penalty: float = 1.2
    penalty_window: int = 50
    min_tokens: int = 200
    max_tokens: int = 9000

    def __post_init__(self):
        if any(type(v) is not int for v in (self.top_k, self.penalty_window, self.min_tokens, self.max_tokens)):
            raise ValueError("Sampling counts must be integers")
        if not all(math.isfinite(v) for v in (self.temperature, self.top_p, self.repetition_penalty)):
            raise ValueError("Sampling values must be finite")
        if not (self.temperature >= 0 and 0 < self.top_p <= 1 and self.top_k > 0 and
                self.repetition_penalty > 0 and self.penalty_window >= 1 and
                0 <= self.min_tokens <= self.max_tokens and self.max_tokens > 0):
            raise ValueError("Invalid sampling settings")


ABC_SAMPLING = Sampling(.7, .9, 30, 1.005, 100, 32, 4096)


class Tokenizer:
    def __init__(self, path):
        import tiktoken
        ranks = {base64.b64decode(t): int(r) for t, r in
                 (line.split() for line in Path(path).read_bytes().splitlines() if line)}
        if len(ranks) != EOD:
            raise ValueError("Expected the original 151643-token qwen.tiktoken")
        pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self._encoding = tiktoken.Encoding("YuE2", pat_str=pattern, mergeable_ranks=ranks, special_tokens={})

    def encode(self, text):
        return self._encoding.encode_ordinary(unicodedata.normalize("NFC", text))

    def decode(self, ids):
        return self._encoding.decode(ids)


def prefix(tokenizer, style, lyrics, cot, abc_ids=None, *, negative=False):
    if cot not in INSTRUCTIONS:
        raise ValueError("cot must be full, melody, or off")
    text = INSTRUCTIONS[cot] if negative else f"{INSTRUCTIONS[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"
    tokens = [EOD] + tokenizer.encode(text)
    if cot == "off":
        return tokens + ([MUSIC_START] if negative else [ABC_START, ABC_END, MUSIC_START])
    if abc_ids is None:
        if negative:
            raise ValueError("Symbolic CFG requires the exact score")
        return tokens + [ABC_START]
    if any(type(i) is not int or not 0 <= i < EOD for i in abc_ids):
        raise ValueError("Invalid ABC token IDs")
    return tokens + [ABC_START] + list(abc_ids) + [ABC_END, MUSIC_START]


def chunk_ranges(frames, prefix_tokens, context=CONTEXT):
    size = min((context - prefix_tokens - 3) // 2, CONTEXT)
    if frames < 1 or size < 1:
        raise ValueError("No acoustic frames or no remaining acoustic context")
    return [(a, min(a + size, frames)) for a in range(0, frames, size)]
