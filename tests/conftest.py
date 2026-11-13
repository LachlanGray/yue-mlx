import importlib.util
from pathlib import Path
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption("--reference", help="Path to an independent upstream YuE2 checkout (not a runtime dependency)")
    parser.addoption("--vae-source", help="Optional original pretrained VAE checkpoint directory")
    parser.addoption("--converted-vae", help="Optional staged checkpoint with the pretrained VAE")


@pytest.fixture(scope="session")
def reference(request):
    root = request.config.getoption("--reference")
    if root is None:
        pytest.skip("Pass --reference /path/to/YuE to run upstream parity")
    root = Path(root).resolve() / "src" / "yue2"
    import torch
    torch.set_num_threads(1)
    # Load reference modules into an isolated namespace, never import its pipeline.
    import types
    package = types.ModuleType("upstream_yue2")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    modules = {}
    for name in ("protocol", "modeling_yue2", "modeling_vae", "nar", "sampling"):
        spec = importlib.util.spec_from_file_location(f"upstream_yue2.{name}", root / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        modules[name] = mod
    return types.SimpleNamespace(**modules)


@pytest.fixture(autouse=True)
def clear_memory():
    yield
    import gc
    import mlx.core as mx
    gc.collect()
    mx.clear_cache()
