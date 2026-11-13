"""Opt-in full pretrained MERT/head parity, using inspected local PyTorch source.

Run after downloading the pinned upstream source into models/hf-cache.
No trust_remote_code download or execution at runtime in the MLX pipeline.
"""
from pathlib import Path
import importlib.util
import sys
import types
import gc

import numpy as np
import torch
import mlx.core as mx
from safetensors.torch import load_file

from yue2_mlx.realaudio import MERT, TokenHead, package
from yue2_mlx.weights import TensorSource, release_memory


def main():
    torch.set_num_threads(4)
    cfg = package()
    folder = Path(cfg['mert'])
    package_name = '_mert_parity_reference'
    pkg = types.ModuleType(package_name); pkg.__path__ = [str(folder)]
    sys.modules[package_name] = pkg
    for name in ('configuration_mert2','modeling_mert2'):
        spec = importlib.util.spec_from_file_location(package_name+'.'+name, folder/(name+'.py'))
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    config = sys.modules[package_name+'.configuration_mert2'].MERT2Config.from_pretrained(str(folder))
    config._attn_implementation='sdpa'
    # CPU reference one model, then fully release it before MLX.
    with torch.device('meta'):
        ref = mod.MERT2Model(config)
    state = load_file(str(folder/'model.safetensors'))
    ref.load_state_dict(state,assign=True)
    ref.layers = torch.nn.ModuleList(list(ref.layers)[:21])
    del state
    ref.eval()
    rng = np.random.default_rng(11)
    x = (rng.normal(size=24000)*.02 + .1*np.sin(2*np.pi*440*np.arange(24000)/24000)).astype('f')
    with torch.no_grad():
        mel = ref.feature_extractor(torch.from_numpy(x[None])).numpy()
        features = ref(torch.from_numpy(x[None])).last_hidden_state[0].numpy().copy()
    del ref; gc.collect()
    source = TensorSource(folder)
    model = MERT({k:source.get(k) for k in source.tensors if not k.startswith('layers.') or int(k.split('.')[1])<=20})
    actual_mel = np.array(model.mel(x)); actual = np.array(model(x))
    print('MERT mel max error:',np.max(np.abs(actual_mel-mel)),flush=True)
    print('MERT features max error:',np.max(np.abs(actual-features)),flush=True)
    np.testing.assert_allclose(actual_mel,mel,atol=2e-4,rtol=2e-4)
    np.testing.assert_allclose(actual,features,atol=3e-3,rtol=3e-3)
    del model, source; release_memory()
    # Real trained head, FP32. No training-script imports (they run training).
    from torch import nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__(); self.inp=nn.Linear(1024,512);self.pos=nn.Parameter(torch.zeros(1,512,512))
            self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(512,8,2048,batch_first=True,norm_first=True,activation='gelu'),8)
            self.norm=nn.LayerNorm(512);self.head=nn.Linear(512,32768)
        def forward(self,x):
            return self.head(self.norm(self.enc(self.inp(x)+self.pos[:,:x.shape[1]])))
    file = str(Path(cfg['tokenizer'])/'tokenizer_head_joint_v4.safetensors')
    ref = Head().eval();ref.load_state_dict(load_file(file))
    h = (features-features.mean(0))/(features.std(0)+1e-5)
    h = np.pad(h,((0,512-len(h)),(0,0)))[None]
    with torch.no_grad(): expected = ref(torch.from_numpy(h)).numpy()
    del ref;gc.collect()
    model = TokenHead(mx.load(file)); actual = np.array(model(mx.array(h)))
    agreement = np.mean(actual.argmax(-1)==expected.argmax(-1))
    print('Trained head max logit error:',np.max(np.abs(actual-expected)),'top1 agreement:',agreement,flush=True)
    np.testing.assert_allclose(actual,expected,atol=3e-3,rtol=3e-3)
    assert agreement >= .99
    print('Pretrained FP32 parity passed.')


if __name__=='__main__':
    main()
