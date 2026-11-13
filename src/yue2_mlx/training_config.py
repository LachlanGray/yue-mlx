from dataclasses import dataclass
import math


@dataclass
class Settings:
    steps: int = 500
    window_frames: int = 256
    save_every: int = 50
    learning_rate_head: float = 1e-4
    learning_rate_lora: float = 5e-5
    learning_rate_io: float = 2e-5
    accumulation: int = 1
    regularizer_songs: int = 4
    regularizer_batch: int = 1
    regularizer_weight: float = 1.
    generated_flow_percent: float = 25.
    regularizer_steps: int = 16
    seed: int = 831001
    train_head: bool = True
    train_nar: bool = True

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data)-set(cls.__dataclass_fields__):
            raise ValueError("Unknown training settings")
        value = cls(**data)
        for key in ('steps','window_frames','save_every','accumulation','regularizer_songs','regularizer_batch','regularizer_steps'):
            if type(getattr(value,key)) is not int or getattr(value,key)<1:
                raise ValueError(f"{key} must be a positive integer")
        if value.window_frames>512:
            raise ValueError("Tokenizer positional weights support at most 512 training frames")
        for key in ('learning_rate_head','learning_rate_lora','learning_rate_io'):
            if not math.isfinite(getattr(value,key)) or getattr(value,key)<=0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(value.regularizer_weight) or value.regularizer_weight<0:
            raise ValueError("Regularizer weight must be finite and nonnegative")
        if not math.isfinite(value.generated_flow_percent) or not 0<=value.generated_flow_percent<=100:
            raise ValueError("Generated flow percentage must be between 0 and 100")
        if type(value.seed) is not int or not 0<=value.seed<2**32:
            raise ValueError("Seed must be a uint32")
        if type(value.train_head) is not bool or type(value.train_nar) is not bool or not (value.train_head or value.train_nar):
            raise ValueError("Select tokenizer head, NAR adapter, or both")
        return value
