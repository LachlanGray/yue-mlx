from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten
from yue2_mlx.model import AR,NAR,Config
from yue2_mlx.training import JointLoss,Settings,straight_through,save_checkpoint,training_head


def tiny():
    c=Config(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
             head_dim=8,intermediate_size=64,vocab_size=64,max_position_embeddings=128)
    ar,nar=AR(c),NAR(c)
    import torch
    from torch import nn
    class Head(nn.Module):
        def __init__(self):
            super().__init__();self.inp=nn.Linear(24,32);self.pos=nn.Parameter(torch.randn(1,512,32)*.01)
            self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,8,128,batch_first=True,norm_first=True,activation='gelu'),8)
            self.norm=nn.LayerNorm(32);self.head=nn.Linear(32,16)
    params=dict(head={k:mx.array(v.detach().numpy()) for k,v in Head().state_dict().items()},lora={},io={})
    for i,layer in enumerate(nar.model.layers):
        for group,names in [('nar_self_attn',('q_proj','k_proj','v_proj','o_proj')),('nar_mlp',('gate_proj','up_proj','down_proj'))]:
            for n in names:
                w=getattr(getattr(layer,group),n).weight
                params['lora'][f'layers.{i}.{group}.{n}.lora_A']=mx.random.normal((4,w.shape[1]))*.01
                params['lora'][f'layers.{i}.{group}.{n}.lora_B']=mx.random.normal((w.shape[0],4))*.01
    for name in ('vae2llm','llm2vae'):
        module=getattr(nar,name)
        params['io'][name+'.weight']=module.weight
        params['io'][name+'.bias']=module.bias
    return ar,nar,params


def test_straight_through_forward_and_gradient():
    x=mx.array([[1.,2.,3.]]);emb=mx.array([[1.,0.],[0.,1.],[2.,2.]])
    np.testing.assert_allclose(np.array(straight_through(x,emb)),[[2,2]],atol=1e-6)
    g=mx.grad(lambda p:mx.sum(straight_through(p,emb)))(x)
    assert np.any(np.abs(np.array(g))>1e-5)


def test_retention_disabled(tmp_path):
    from yue2_mlx.training import prepare
    settings=Settings.parse(dict(generated_flow_percent=0,regularizer_weight=0))
    # No supplementary generation/model access when both retention paths are off.
    assert prepare(None,[],None,tmp_path,settings,lambda:False)==([],[])
    assert Settings.parse({}).generated_flow_percent==25
    for value in (-1,101,float('nan')):
        with pytest.raises(ValueError):Settings.parse(dict(generated_flow_percent=value))


@pytest.mark.parametrize('quantized',[False,True])
def test_joint_gradients_and_frozen_base(quantized):
    mx.random.seed(12);ar,nar,params=tiny();loss=JointLoss(ar,nar,16,16,0)
    if quantized:
        import mlx.nn as nn
        for model in (ar,nar):
            nn.quantize(model,group_size=32,bits=8,class_predicate=lambda n,m:isinstance(m,nn.Linear) and n.startswith('model.layers.') and n.endswith('_proj'))
    x=mx.random.normal((6,24));z=mx.random.normal((6,64));noise=mx.random.normal(z.shape)
    before=np.array(ar.model.layers[0].mlp.up_proj.weight)
    def objective(p):return loss.flow(p,x,z,mx.array([1,2]),.3,noise)
    value,grads=mx.value_and_grad(objective)(params);mx.eval(value,grads)
    assert np.isfinite(value.item())
    for group in params:
        assert all(np.isfinite(np.array(g)).all() for g in grads[group].values())
        assert sum(float(mx.sum(mx.abs(g)).item()) for g in grads[group].values())>0
    np.testing.assert_array_equal(np.array(ar.model.layers[0].mlp.up_proj.weight),before)
    # Training dropout is deterministic for the supplied key and differentiable.
    key=mx.random.key(123)
    a=training_head(params['head'],x[None],key);b=training_head(params['head'],x[None],key)
    np.testing.assert_array_equal(np.array(a),np.array(b))
    cg=mx.grad(lambda p:loss.regularize({'head':p},x,mx.zeros((6,),mx.int32),key))(params['head'])
    assert np.isfinite(np.array(cg['head.weight'])).all()


def test_checkpoint_schema_and_registry(tmp_path):
    from yue2_mlx.web import Studio
    _,_,params=tiny()
    source=tmp_path/'base';source.mkdir();(source/'manifest.json').write_text(json.dumps(dict(format='yue2-realaudio-v4',mert='unused',tokenizer='unused')))
    studio=Studio(tmp_path/'jobs',source,source,tokenizer_model=source)
    identifier='a'*32;studio.jobs[identifier]=dict(id=identifier,kind='train')
    checkpoints=studio.root/identifier/'output/checkpoints'
    partial=checkpoints/'step-00000099';partial.mkdir(parents=True)
    (partial/'tokenizer_head_joint_v4.safetensors').write_bytes(b'partial')
    assert len(studio.tokenizers())==1
    saved=save_checkpoint(checkpoints,1,params,source,Settings(),.25,'My model')
    choices=studio.tokenizers();assert len(choices)==2 and choices[1]['name']=='My model · step 1'
    assert studio.tokenizer_path(choices[1]['id'])==str(saved)
    weights=mx.load(str(saved/'nar_lora_joint_v4.safetensors'))
    assert set(weights)==set(params['lora'])|set(params['io'])
    with pytest.raises(FileExistsError):save_checkpoint(checkpoints,1,params,source,Settings(),.25,'Again')


def test_training_queue_dataset_snapshot(tmp_path,monkeypatch):
    from yue2_mlx.web import Studio
    model=tmp_path/'model';model.mkdir();(model/'manifest.json').write_text('{}')
    studio=Studio(tmp_path/'jobs',model,model,tokenizer_model=model)
    monkeypatch.setattr(studio,'launch_next',lambda:None)
    for i in (1,2):(studio.root/'uploads'/f'{i:032x}').write_bytes(b'audio')
    studio.set_training_songs([dict(id=f'{i:032x}',name=f'song{i}') for i in (1,2)])
    job=studio.start(dict(kind='train',name='My model',parameters=dict(steps=2)))
    studio.set_training_songs([])
    assert len(job['settings']['songs'])==2 and job['status']=='queued'
    assert all(Path(s['path']).exists() for s in job['settings']['songs'])
    assert 'yue2_mlx.training' in job['command']
    studio.cancel(job['id']);assert studio.jobs[job['id']]['status']=='cancelled'
    with pytest.raises(ValueError):studio.start(dict(kind='train'))
    with pytest.raises(ValueError):Settings.parse(dict(window_frames=513))
    with pytest.raises(ValueError):Settings.parse(dict(train_head=False,train_nar=False))


@pytest.mark.parametrize('stop_after_first',[False,True])
def test_optimizer_loop_and_stop_checkpoint(tmp_path,monkeypatch,stop_after_first):
    import signal,os
    from types import SimpleNamespace
    import yue2_mlx.training as tr
    ar,nar,params=tiny()
    source=tmp_path/'base';source.mkdir();(source/'manifest.json').write_text(json.dumps(dict(format='yue2-realaudio-v4',mert='unused',tokenizer='unused')))
    real=tmp_path/'real';minted=tmp_path/'minted'
    for folder in (real,minted):
        folder.mkdir();rng=np.random.default_rng(42)
        np.save(folder/'features.npy',rng.normal(size=(12,24)).astype('f'))
        np.save(folder/'latents.npy',rng.normal(size=(12,64)).astype('f'))
    np.save(minted/'tokens.npy',np.arange(12,dtype=np.int32))
    (minted/'prefix.json').write_text('[1,2]')
    class Pipe:
        def __init__(self,*a,**kw):self.checkpoint=SimpleNamespace(load=lambda name,**kw:ar if name=='ar' else nar)
        def plan(self,**kw):return dict(prefix=[1,2])
    monkeypatch.setattr(tr,'Pipeline',Pipe)
    monkeypatch.setattr(tr,'prepare',lambda *a:([real],[minted]))
    monkeypatch.setattr(tr,'load_parameters',lambda *a:params)
    original_loss=tr.JointLoss
    monkeypatch.setattr(tr,'JointLoss',lambda a,n:original_loss(a,n,16,16,0))
    original_save=tr.save_checkpoint
    def save(*a,**kw):
        result=original_save(*a,**kw)
        if stop_after_first and a[1]==1:os.kill(os.getpid(),signal.SIGTERM)
        return result
    monkeypatch.setattr(tr,'save_checkpoint',save)
    config=tmp_path/'request.json';config.write_text(json.dumps(dict(parameters=dict(steps=2,save_every=1,window_frames=4),
        tokenizer_model=str(source),name='Test',songs=[dict(path='unused')],cache='unused')))
    before=np.array(params['head']['head.weight'])
    tr.train(config,tmp_path/'output','unused')
    report=json.loads((tmp_path/'output/result.json').read_text())
    assert report['step']==(1 if stop_after_first else 2)
    assert report['stopped']==stop_after_first
    weights=mx.load(str(tmp_path/'output/checkpoints'/report['checkpoint']/'tokenizer_head_joint_v4.safetensors'))
    assert not np.array_equal(before,np.array(weights['head.weight']))
    assert len(list((tmp_path/'output/checkpoints').glob('*/manifest.json')))==report['step']+1
    if stop_after_first:
        # Same base weights/data: interruption must not reset Adam, RNG or schedule.
        monkeypatch.setattr(tr,'save_checkpoint',original_save)
        tr.train(config,tmp_path/'uninterrupted','unused')
        checkpoint=tmp_path/'output/checkpoints'/report['checkpoint']
        request=json.loads(config.read_text());request['resume_checkpoint']=str(checkpoint)
        config.write_text(json.dumps(request))
        def load(root):
            if str(root)==str(source):return params
            folder=Path(root)
            head=mx.load(str(folder/'tokenizer_head_joint_v4.safetensors'))
            other=mx.load(str(folder/'nar_lora_joint_v4.safetensors'))
            return dict(head=head,lora={k:v for k,v in other.items() if '.lora_' in k},io={k:v for k,v in other.items() if '.lora_' not in k})
        monkeypatch.setattr(tr,'load_parameters',load)
        tr.train(config,tmp_path/'resumed','unused')
        for filename in ('tokenizer_head_joint_v4.safetensors','nar_lora_joint_v4.safetensors','optimizer.safetensors'):
            expected=mx.load(str(tmp_path/'uninterrupted/checkpoints/step-00000002'/filename))
            actual=mx.load(str(tmp_path/'resumed/checkpoints/step-00000002'/filename))
            assert expected.keys()==actual.keys()
            # Reloading can change MLX evaluation/fusion rounding by a few ulps.
            for k in expected:np.testing.assert_allclose(np.array(actual[k]),np.array(expected[k]),rtol=1e-5,atol=1e-10,err_msg=k)
        history=[json.loads(line) for line in (tmp_path/'resumed/metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history]==[1,2]
        assert all('validation_loss' in r for r in history)


def test_resume_queue_and_metrics(tmp_path,monkeypatch):
    from yue2_mlx.web import Studio
    model=tmp_path/'model';model.mkdir();(model/'manifest.json').write_text('{}')
    studio=Studio(tmp_path/'jobs',model,model,tokenizer_model=model)
    monkeypatch.setattr(studio,'launch_next',lambda:None)
    song='b'*32;(studio.root/'uploads'/song).write_bytes(b'audio')
    studio.set_training_songs([dict(id=song,name='Original')])
    job=studio.start(dict(kind='train',parameters=dict(steps=10)))
    studio.cancel(job['id']);studio.set_training_songs([])
    output=studio.root/job['id']/'output';folder=output/'checkpoints/step-00000003';folder.mkdir(parents=True)
    (output/'metrics.jsonl').write_text('{"step":3,"flow_loss":0.5}\n{"step":')
    manifest=dict(training=dict(optimizer_resume=True,step=3,settings=job['settings']['parameters']))
    (folder/'manifest.json').write_text(json.dumps(manifest))
    assert not studio.training_progress(job['id'])['resumable']
    for name in ('resume.json','optimizer.safetensors','tokenizer_head_joint_v4.safetensors','nar_lora_joint_v4.safetensors'):(folder/name).write_text('{}')
    progress=studio.training_progress(job['id']);assert progress['resumable'] and len(progress['metrics'])==1
    resumed=studio.start(dict(kind='train',resume_job=job['id']))
    assert resumed['settings']['songs']==job['settings']['songs']
    assert resumed['settings']['parameters']==job['settings']['parameters']
    assert resumed['settings']['resume_checkpoint']==str(folder)
    with pytest.raises(ValueError,match='already'):studio.start(dict(kind='train',resume_job=job['id']))
    with pytest.raises(ValueError):studio.start(dict(kind='train',resume_job='../bad'))
