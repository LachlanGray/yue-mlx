"""Layer parity, prediction window coverage, and adapter arithmetic."""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from torch import nn as tnn

from yue2_mlx.realaudio import TokenHead, LoRALinear, MERT


def test_head_parity():
    torch.manual_seed(17)
    class Head(tnn.Module):
        def __init__(self):
            super().__init__()
            self.inp = tnn.Linear(24, 32)
            self.pos = tnn.Parameter(torch.randn(1, 512, 32) * .02)
            self.enc = tnn.TransformerEncoder(tnn.TransformerEncoderLayer(
                32, 8, 128, batch_first=True, norm_first=True, activation='gelu'), 8)
            self.norm = tnn.LayerNorm(32)
            self.head = tnn.Linear(32, 71)
        def forward(self, x):
            return self.head(self.norm(self.enc(self.inp(x) + self.pos[:, :x.shape[1]])))
    ref = Head().eval()
    model = TokenHead({k: mx.array(v.detach().numpy()) for k, v in ref.state_dict().items()})
    x = np.random.default_rng(2).normal(size=(1, 37, 24)).astype(np.float32)
    with torch.no_grad():
        expected = ref(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(np.array(model(mx.array(x))), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('frames', [1, 511, 512, 513, 768, 769, 1024, 1503])
def test_prediction_window_coverage(frames):
    class Head(TokenHead):
        def __call__(self, x):
            return mx.broadcast_to(mx.array([0., 1.]), (1, x.shape[1], 2))
    result = Head({}).predict(np.random.default_rng(3).normal(size=(frames, 2)))
    np.testing.assert_array_equal(result, np.ones(frames, np.int32))


def test_lora_on_quantized_base():
    base = nn.Linear(64, 32, bias=False).to_quantized(group_size=32, bits=8)
    a = mx.random.normal((4, 64)) * .02
    b = mx.random.normal((32, 4)) * .02
    x = mx.random.normal((1, 5, 64))
    result = LoRALinear(base, a, b)(x)
    expected = np.array(base(x)) + (np.array(x) @ np.array(a).T) @ np.array(b).T
    np.testing.assert_allclose(np.array(result), expected, atol=1e-6)


def test_convnext_parity():
    # Match one published ConvNext block including temporal global-response norm.
    rng = np.random.default_rng(9)
    width = 8
    w = {'c.depthwise_block.1.weight': rng.normal(size=(width,1,7)).astype('f')*.1,
         'c.depthwise_block.1.bias': np.zeros(width, 'f')}
    for p, out, inp in [('1', 4*width, width), ('4', width, 4*width)]:
        w[f'c.pointwise_block.{p}.weight'] = rng.normal(size=(out,inp)).astype('f')*.1
        w[f'c.pointwise_block.{p}.bias'] = np.zeros(out, 'f')
    w.update({'c.pointwise_block.0.weight':np.ones(width,'f'), 'c.pointwise_block.0.bias':np.zeros(width,'f'),
              'c.pointwise_block.3.weight':rng.normal(size=(1,1,4*width)).astype('f'),
              'c.pointwise_block.3.bias':rng.normal(size=(1,1,4*width)).astype('f')})
    x = rng.normal(size=(1,31,width)).astype('f')
    tw = {k:torch.tensor(v) for k,v in w.items()}
    f = torch.nn.functional
    h = f.conv1d(torch.tensor(x).transpose(1,2),tw['c.depthwise_block.1.weight'],tw['c.depthwise_block.1.bias'],padding=3,groups=width).transpose(1,2)
    h = f.layer_norm(h,(width,),tw['c.pointwise_block.0.weight'],tw['c.pointwise_block.0.bias'],1e-6)
    h = f.gelu(f.linear(h,tw['c.pointwise_block.1.weight'],tw['c.pointwise_block.1.bias']))
    g = torch.norm(h,p=2,dim=1,keepdim=True); n = g/(g.mean(-1,keepdim=True)+1e-6)
    h = h + tw['c.pointwise_block.3.weight']*h*n + tw['c.pointwise_block.3.bias']
    expected = torch.tensor(x)+f.linear(h,tw['c.pointwise_block.4.weight'],tw['c.pointwise_block.4.bias'])
    model = MERT({k:mx.array(v) for k,v in w.items()})
    np.testing.assert_allclose(np.array(model.convnext(mx.array(x),'c')),expected.numpy(),atol=2e-5,rtol=2e-5)


def test_web_semantic_queue(tmp_path, monkeypatch):
    from yue2_mlx.web import Studio
    model = tmp_path/'model'; model.mkdir(); (model/'manifest.json').write_text('{}')
    studio = Studio(tmp_path/'jobs', model, model, tokenizer_model=model)
    monkeypatch.setattr(studio, 'launch_next', lambda: None)
    upload = 'a'*32; (studio.root/'uploads'/upload).write_bytes(b'fixture')
    for enabled in (True,False):
        job = studio.start(dict(kind='semantic-audio',upload=upload,style='piano',lyrics='',nar_adapter=enabled,steps=12,seed=42))
        assert job['status']=='queued'
        assert job['settings']['nar_adapter'] is enabled
        assert ('--no-nar-adapter' in job['command']) is (not enabled)
        assert '--tokenizer-model' in job['command'] and '--lyrics-file' in job['command']
        assert json.loads((studio.root/job['id']/'job.json').read_text())['settings']==job['settings']
    assert studio.snapshot()['semantic_audio_ready']
    with pytest.raises(ValueError):
        studio.start(dict(kind='semantic-audio',upload=upload,nar_adapter='false'))
