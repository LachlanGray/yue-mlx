"""Independent MLX implementation of the YuE2 inference architecture."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "Pipeline":
        from .pipeline import Pipeline
        return Pipeline
    raise AttributeError(name)
