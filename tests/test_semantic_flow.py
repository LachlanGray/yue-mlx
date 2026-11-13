"""Combined semantic conditioning / source initialization / reference plumbing."""
from contextlib import contextmanager
from types import SimpleNamespace
import json
import numpy as np
import pytest
import soundfile as sf
import mlx.core as mx
from yue2_mlx.pipeline import Pipeline


@pytest.mark.parametrize('mode,t,amount,effective', [('fresh',.2,.3,1),('mix',.3,None,.3),('mix',.3,.8,.8),('direct',.2,.7,0),('anchored',.3,.1,.1),('anchored',0,.8,0)])
def test_combined(tmp_path,monkeypatch,mode,t,amount,effective):
    from yue2_mlx import realaudio
    monkeypatch.setattr(realaudio,'package',lambda _: {})
    monkeypatch.setattr(realaudio,'extract',lambda *a: np.arange(7,dtype=np.int32))
    pipe=Pipeline.__new__(Pipeline)
    pipe.checkpoint=SimpleNamespace(manifest={'model_config':{'latent_dim':64,'max_position_embeddings':30},'vae_config':{'sample_rate':48000}})
    pipe.memory_events=[];pipe.progress=None;pipe.query_chunk=16;pipe.realaudio_adapter=None
    pipe.plan=lambda **kw:dict(prefix=[1,2],seed=kw['seed'])
    baseline=np.ones((6,64),np.float32)*.25
    pipe.encode=lambda *a:baseline.copy()
    pipe.decode=lambda z,p:sf.write(p,np.zeros((100,2)),48000)
    calls=[]
    class AR:
        def condition(self,ids,**kw):
            calls.append(('ar',np.array(ids)));return []
    class NAR:
        def reference_cache(self,state,*a,**kw):calls.append(('reference',kw));return {'test':True}
        def solve(self,state,cache,length,**kw):calls.append(('solve',kw));return state
    @contextmanager
    def stage(name,**kw):yield AR() if name=='ar' else NAR()
    pipe._stage=stage
    source=tmp_path/'source.wav';sf.write(source,np.ones((12000,2))*.1,48000)
    out=tmp_path/'out'
    pipe.semantic_audio(source,out,mode=mode,start_time=t,noise_amount=amount,reference_path=source,
                        reference_placement='audio',seed=17)
    noise=np.array(mx.random.normal(baseline.shape,key=mx.random.key(17)))
    np.testing.assert_allclose(np.load(out/'initial-latents.npy'),(1-effective)*baseline+effective*noise,atol=1e-6)
    report=json.loads((out/'result.json').read_text())
    assert report['semantic_frames']==6 and report['effective_noise_amount']==effective
    assert pipe.realaudio_adapter is None
    if t==0 and mode!='fresh':assert not calls
    else:
        ar=[c[1] for c in calls if c[0]=='ar'][0]
        assert 151853 in ar and 151858 in ar
        solve=[c[1] for c in calls if c[0]=='solve'][0]
        assert solve['start_time']==(1 if mode=='fresh' else t)
        assert solve['reference']=={'test':True}
        assert ('anchor_source' in solve)==(mode=='anchored')
        assert [c[1]['placement'] for c in calls if c[0]=='reference']==['audio']
