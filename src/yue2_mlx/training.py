"""Local audio-only joint tokenizer/NAR adaptation. No PyTorch runtime.

This is an adapted recipe, not a verbatim reproduction of joint.py: a small
automatically minted corpus, hard-token CE regularization (not neighbor-soft CE),
microbatching, and held-out windows when only one source song is supplied.
"""
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten
import numpy as np
import soundfile as sf

from .model import audio_positions, rotary
from .pipeline import Pipeline
from .protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END, Sampling
from .realaudio import Ops, TokenHead, package, extract_features
from .weights import release_memory


from .training_config import Settings


def straight_through(logits, embeddings):
    probabilities = mx.softmax(logits.astype(mx.float32), axis=-1)
    soft = probabilities @ embeddings.astype(mx.float32)
    hard = embeddings[mx.argmax(probabilities, axis=-1)].astype(mx.float32)
    return soft + mx.stop_gradient(hard-soft)


def training_head(weights, x, key=None):
    """Published head with training dropout; inference remains exactly TokenHead."""
    if key is None:
        return TokenHead(weights)(x)
    op = Ops(weights)
    x = op.linear(x, 'inp')+weights['pos'][:,:x.shape[1]]
    width=x.shape[-1]
    def drop(h,k):
        return mx.where(mx.random.bernoulli(.9,h.shape,key=k),h/.9,0)
    for i in range(8):
        p=f'enc.layers.{i}'
        key,*keys=mx.random.split(key,5)
        h=op.norm(x,p+'.norm1')
        h=h@weights[p+'.self_attn.in_proj_weight'].T+weights[p+'.self_attn.in_proj_bias']
        q,k,v=[a.reshape(1,-1,8,width//8).transpose(0,2,1,3) for a in mx.split(h,3,axis=-1)]
        scores=mx.softmax((q@k.swapaxes(-1,-2))*((width//8)**-.5),axis=-1)
        h=(drop(scores,keys[0])@v).transpose(0,2,1,3).reshape(x.shape)
        x=x+drop(op.linear(h,p+'.self_attn.out_proj'),keys[1])
        h=drop(nn.gelu(op.linear(op.norm(x,p+'.norm2'),p+'.linear1')),keys[2])
        x=x+drop(op.linear(h,p+'.linear2'),keys[3])
    return op.linear(op.norm(x,'norm'),'head')


def attend(q,k,v,causal=False):
    return mx.fast.scaled_dot_product_attention(q,k,v,scale=q.shape[-1]**-.5,
                                                mask='causal' if causal else None).transpose(0,2,1,3)


def ar_block(layer,x):
    q,k,v=layer.self_attn.project(layer.input_layernorm(x))
    h=attend(q,k,v,True).reshape(x.shape)
    x=x+layer.self_attn.o_proj(h)
    return x+layer.mlp(layer.post_attention_layernorm(x)),k,v


def nar_block(layer, params, x, ak, av, offset):
    c=layer.nar_self_attn._config
    def linear(h, group, name):
        base=getattr(getattr(layer,group),name)
        a,b=params[f'{group}.{name}.lora_A'],params[f'{group}.{name}.lora_B']
        return base(h)+((h.astype(mx.float32)@a.T)@b.T).astype(h.dtype)
    h=layer.nar_input_layernorm(x); b,t,_=h.shape
    q=layer.nar_self_attn.q_norm(linear(h,'nar_self_attn','q_proj').reshape(b,t,c.num_attention_heads,c.head_dim))
    k=layer.nar_self_attn.k_norm(linear(h,'nar_self_attn','k_proj').reshape(b,t,c.num_key_value_heads,c.head_dim))
    v=linear(h,'nar_self_attn','v_proj').reshape(b,t,c.num_key_value_heads,c.head_dim).transpose(0,2,1,3)
    q,k=rotary(q.transpose(0,2,1,3),offset,c.rope_theta),rotary(k.transpose(0,2,1,3),offset,c.rope_theta)
    h=attend(q,mx.concatenate((ak,k),axis=2),mx.concatenate((av,v),axis=2)).reshape(x.shape)
    x=x+linear(h,'nar_self_attn','o_proj')
    h=layer.nar_pre_mlp_layernorm(x)
    return x+linear(nn.silu(linear(h,'nar_mlp','gate_proj'))*linear(h,'nar_mlp','up_proj'),'nar_mlp','down_proj')


class JointLoss:
    def __init__(self, ar, nar, codec_offset=CODEC_OFFSET, codec_size=CODEC_SIZE, music_end=MUSIC_END):
        self.ar,self.nar=ar,nar
        self.embeddings=ar.model.embed_tokens.weight[codec_offset:codec_offset+codec_size]
        self.music_end=music_end

    def flow(self,params,features,latents,prefix,t,noise,*,tokens=None,key=None):
        logits=training_head(params['head'],features[None],key)[0] if tokens is None else None
        emb=straight_through(logits,self.embeddings) if tokens is None else self.embeddings[tokens]
        pre=self.ar.model.embed_tokens(prefix)
        end=self.ar.model.embed_tokens(mx.array([self.music_end]))
        x=mx.concatenate((pre,emb.astype(pre.dtype),end),axis=0)[None]
        length=x.shape[1]
        if length+len(latents)+2>self.nar._config.max_position_embeddings:
            raise ValueError("Training window and prompt exceed model context")
        cache=[]
        for layer in self.ar.model.layers:
            x,k,v=mx.checkpoint(partial(ar_block,layer))(x)
            cache.append((k,v))
        xt=t*noise+(1-t)*latents
        io=Ops(params['io']);dtype=self.nar.model.norm.weight.dtype
        h=io.linear(mx.pad(xt,((1,1),(0,0)))[None].astype(mx.float32),'vae2llm').astype(dtype)
        h=h+self.nar.time_embedder(mx.array([t],dtype=dtype),dtype)[None]
        h=h+audio_positions(len(latents)+2,self.nar._config,dtype)[None]
        for i,(layer,(ak,av)) in enumerate(zip(self.nar.model.layers,cache)):
            p={k.removeprefix(f'layers.{i}.'):v for k,v in params['lora'].items() if k.startswith(f'layers.{i}.')}
            h=mx.checkpoint(partial(nar_block,layer))(p,h,ak,av,length)
        prediction=io.linear(self.nar.model.norm(h)[0,1:-1].astype(mx.float32),'llm2vae')
        return mx.mean((prediction-(noise-latents))**2)

    def regularize(self,params,features,tokens,key=None):
        logits=training_head(params['head'],features[None],key)[0]
        return nn.losses.cross_entropy(logits,tokens,reduction='mean')


def load_parameters(root):
    config=package(root);folder=Path(config['tokenizer'])
    head=mx.load(str(folder/'tokenizer_head_joint_v4.safetensors'))
    nar=mx.load(str(folder/'nar_lora_joint_v4.safetensors'))
    return dict(head=head,lora={k:v for k,v in nar.items() if '.lora_' in k},
                io={k:v for k,v in nar.items() if '.lora_' not in k})


def save_resume_state(folder, state):
    arrays={}
    def pack(value):
        if isinstance(value,mx.array):
            key=str(len(arrays));arrays[key]=value;return {'array':key}
        if isinstance(value,dict):return {'dict':[[k,pack(v)] for k,v in value.items()]}
        if isinstance(value,(tuple,list)):return {'list':[pack(v) for v in value]}
        return {'value':value}
    metadata=pack(state)
    mx.save_safetensors(str(folder/'optimizer.safetensors'),arrays)
    (folder/'resume.json').write_text(json.dumps(metadata))


def load_resume_state(folder):
    folder=Path(folder);arrays=mx.load(str(folder/'optimizer.safetensors'))
    def unpack(value):
        if 'array' in value:return arrays[value['array']]
        if 'dict' in value:return {k:unpack(v) for k,v in value['dict']}
        if 'list' in value:return [unpack(v) for v in value['list']]
        return value['value']
    return unpack(json.loads((folder/'resume.json').read_text()))


def save_checkpoint(root, step, params, source, settings, metric, name, *, resume_state=None):
    """Immutable complete pair: publish manifest last, never expose partial writes."""
    if not math.isfinite(metric):raise FloatingPointError('Nonfinite validation loss; previous checkpoints retained')
    if not mx.all(mx.stack([mx.all(mx.isfinite(v)) for group in params.values() for v in group.values()])).item():
        raise FloatingPointError('Nonfinite checkpoint weights; previous checkpoints retained')
    folder=Path(root)/f'step-{step:08d}'
    folder.mkdir(parents=True,exist_ok=False)
    mx.save_safetensors(str(folder/'tokenizer_head_joint_v4.safetensors'),params['head'])
    mx.save_safetensors(str(folder/'nar_lora_joint_v4.safetensors'),{**params['lora'],**params['io']})
    if resume_state is not None:save_resume_state(folder,resume_state)
    manifest=dict(package(source),tokenizer=str(folder.resolve()),name=f'{name} · step {step}',
                  training=dict(step=step,settings=asdict(settings),metric=metric,
                                recipe='audio-only-local-joint-v1',optimizer_resume=resume_state is not None),parent=str(Path(source).resolve()))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return folder


def normalized(features):
    features=features.astype(np.float32)
    return (features-features.mean(0))/(features.std(0)+1e-5)


def prepare(pipe,songs,source,cache,settings,stop):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    def one(file,destination):
        if (destination/'ready.json').exists():return destination
        destination.mkdir(parents=True,exist_ok=True)
        audio,rate=sf.read(file,dtype='float32',always_2d=True)
        features=extract_features(audio,rate,source,pipe._report)
        audio48=pipe._read_reference(file)
        latents=pipe.encode(audio48,48000)
        n=min(len(features),len(latents))
        if n<2:raise ValueError('Each training song needs at least two acoustic frames')
        np.save(destination/'features.npy',normalized(features[:n]))
        np.save(destination/'latents.npy',latents[:n])
        (destination/'ready.json').write_text(json.dumps(dict(frames=n)))
        return destination
    real=[]
    for i,song in enumerate(songs):
        if stop():return [],[]
        pipe._report('prepare_song',i+1,len(songs))
        file=Path(song['path'])
        digest=hashlib.sha256()
        with file.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
        digest.update(json.dumps(dict(mert=package(source)['mert_revision'],vae=pipe.checkpoint.manifest['sources'],recipe=1),sort_keys=True).encode())
        real.append(one(file,cache/('song-'+digest.hexdigest())))
    minted=[]
    if not ((settings.train_nar and settings.generated_flow_percent>0) or
            (settings.train_head and settings.regularizer_weight>0)):
        return real,minted
    from dataclasses import replace
    styles=['instrumental, acoustic piano, warm strings','instrumental, electric guitar, rock drums, bass',
            'instrumental, jazz piano, upright bass, brushed drums','instrumental, electronic synths, dance beat']
    for i in range(settings.regularizer_songs):
        if stop():return [],[]
        pipe._report('prepare_regularizer',i+1,settings.regularizer_songs)
        key=hashlib.sha256(json.dumps(dict(model=pipe.checkpoint.manifest['sources'],frames=settings.window_frames,
            seed=settings.seed+i,style=styles[i%len(styles)],steps=settings.regularizer_steps,recipe=1),sort_keys=True).encode()).hexdigest()
        folder=cache/('minted-'+key)
        generation=folder/'generation'
        completed=folder/'generation-path.json'
        if completed.exists():
            generation=folder/json.loads(completed.read_text())
            if generation.resolve().parent!=folder.resolve():raise ValueError('Invalid cached generation path')
        if not (generation/'result.json').exists():
            # An interrupted generation is retained; retry in a new child.
            if generation.exists():generation=folder/f'generation-{time.time_ns()}'
            folder.mkdir(parents=True,exist_ok=True)
            pipe.generate(generation,style=styles[i%len(styles)],lyrics='',cot='off',seed=(settings.seed+i)%2**32,
                          steps=settings.regularizer_steps,
                          sampling=replace(Sampling(),min_tokens=max(2,settings.window_frames),max_tokens=max(2,settings.window_frames)))
            completed.write_text(json.dumps(generation.name))
        prepared=one(generation/'audio.flac',folder/'prepared')
        semantic=json.loads((generation/'semantic.json').read_text())
        np.save(prepared/'tokens.npy',np.asarray(semantic['tokens'],np.int32))
        # Use the original generated latent target, not a re-encoded approximation.
        np.save(prepared/'latents.npy',np.load(generation/'latents.npy',allow_pickle=False))
        (prepared/'prefix.json').write_text(json.dumps(semantic['plan']['prefix']))
        minted.append(prepared)
    return real,minted


def train(config_path,output,model,*,memory_gib=10):
    request=json.loads(Path(config_path).read_text());settings=Settings.parse(request['parameters'])
    source=request['tokenizer_model'];name=request['name'];songs=request['songs']
    if not songs:raise ValueError('Add at least one song')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    stopping=[False]
    old_handler=signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    def progress(phase,done=0,total=0):print(phase+(f': {done}/{total}' if total else ''),flush=True)
    pipe=Pipeline(model,memory_gib=memory_gib,progress=progress)
    params=None;step=0;last_saved=-1;metric=None
    started=time.perf_counter()
    try:
        resumed=load_resume_state(request['resume_checkpoint']) if request.get('resume_checkpoint') else None
        if resumed and (asdict(Settings.parse(resumed['settings']))!=asdict(settings) or resumed['model']!=str(Path(model).resolve()) or resumed['songs']!=songs):
            raise ValueError('Resume requires the original settings, model and dataset')
        real,minted=prepare(pipe,songs,source,request['cache'],settings,lambda:stopping[0])
        if stopping[0]:return
        # Memory-map dataset instead of retaining all songs in unified memory.
        def record(p):
            x=dict(features=np.load(p/'features.npy',mmap_mode='r'),latents=np.load(p/'latents.npy',mmap_mode='r'))
            if (p/'tokens.npy').exists():
                x['tokens']=np.load(p/'tokens.npy',mmap_mode='r');x['prefix']=json.loads((p/'prefix.json').read_text())
            return x
        real,minted=[record(p) for p in real],[record(p) for p in minted]
        # A whole held-out song when possible; otherwise disjoint last-quarter frames.
        if len(real)>1:hold=real[-1];real=real[:-1]
        else:
            item=real[0];n=min(len(item['features']),len(item['latents']));split=max(1,int(n*.75))
            hold={k:v[split:n] for k,v in item.items()};real=[{k:v[:split] for k,v in item.items()}]
        prefix=pipe.plan(style='',lyrics='',cot='off',seed=settings.seed)['prefix']
        progress('load_training_ar');ar=pipe.checkpoint.load('ar',head=False)
        progress('load_training_nar');nar=pipe.checkpoint.load('nar')
        if nar._config.timestep_shift!=1:raise ValueError('Training requires timestep_shift=1')
        params=load_parameters(request.get('resume_checkpoint',source));mx.eval(params)
        loss=JointLoss(ar,nar)
        active=[k for k in params if (k=='head' and settings.train_head) or (k!='head' and settings.train_nar)]
        frozen={k:tree_map(mx.stop_gradient,v) for k,v in params.items() if k not in active}
        trainable={k:params[k] for k in active}
        rates=dict(head=settings.learning_rate_head,lora=settings.learning_rate_lora,io=settings.learning_rate_io)
        optimizers={k:optim.AdamW(learning_rate=rates[k],betas=(.9,.95),weight_decay=.05 if k=='head' else 0.) for k in active}
        rng=np.random.default_rng(settings.seed)
        if resumed:
            step=resumed['step'];rng.bit_generator.state=resumed['rng']
            for k in active:optimizers[k].state=resumed['optimizers'][k]
        def window(item,random=True):
            n=min(len(v) for k,v in item.items() if k in ('features','latents','tokens'))
            size=min(settings.window_frames,n)
            start=int(rng.integers(0,n-size+1)) if random else 0
            return mx.array(np.array(item['features'][start:start+size])),mx.array(np.array(item['latents'][start:start+size])),\
                   mx.array(np.array(item['tokens'][start:start+size])) if 'tokens' in item else None
        hf,hz,_=window(hold,False)
        hn=mx.random.normal(hz.shape,key=mx.random.key(123));mx.eval(hf,hz,hn)
        def evaluate():
            losses=[loss.flow(params,hf,hz,mx.array(prefix),t,hn) for t in (.2,.5,.8)]
            return float(mx.mean(mx.stack(losses)).item())
        metric=resumed['metric'] if resumed else evaluate()
        if not math.isfinite(metric):raise FloatingPointError('Nonfinite initial validation loss')
        best=resumed['best'] if resumed else metric
        best_checkpoint=resumed.get('best_checkpoint') if resumed else None
        if not best_checkpoint:best_checkpoint=str((output/'checkpoints'/f'step-{step:08d}').resolve())
        history=resumed['history'] if resumed else []
        def checkpoint():
            state=dict(step=step,settings=asdict(settings),model=str(Path(model).resolve()),songs=songs,
                       rng=rng.bit_generator.state,optimizers={k:o.state for k,o in optimizers.items()},
                       metric=metric,best=best,best_checkpoint=best_checkpoint,history=history)
            return save_checkpoint(output/'checkpoints',step,params,source,settings,metric,name,resume_state=state)
        checkpoint();last_saved=step
        (output/'best.json').write_text(json.dumps(dict(checkpoint=best_checkpoint,validation_loss=best)))
        (output/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
        print(f'validation step {step} loss {metric:.6f}',flush=True)
        with (output/'metrics.jsonl').open('a') as log:
            for step in range(step+1,settings.steps+1):
                if stopping[0]:step-=1;break
                accum=None;total=0.;ce_total=0.
                for micro in range(settings.accumulation):
                    use_minted=settings.train_nar and rng.random()<settings.generated_flow_percent/100
                    item=minted[int(rng.integers(len(minted)))] if use_minted else real[int(rng.integers(len(real)))]
                    f,z,tokens=window(item)
                    t=float(np.clip(rng.beta(2,2),.05,.95))
                    key=mx.random.key(int(rng.integers(2**32)))
                    noise=mx.random.normal(z.shape,key=key)
                    def objective(p):
                        return loss.flow({**frozen,**p},f,z,mx.array(item.get('prefix',prefix)),t,noise,
                                         tokens=tokens if use_minted else None,key=key if settings.train_head else None)
                    value,grads=mx.value_and_grad(objective)(trainable)
                    mx.eval(value,grads);total+=float(value.item())/settings.accumulation
                    grads=tree_map(lambda g:g/settings.accumulation,grads)
                    accum=grads if accum is None else tree_map(lambda a,b:a+b,accum,grads)
                    mx.eval(accum)
                    del grads,value
                if settings.train_head and settings.regularizer_weight:
                    for _ in range(settings.regularizer_batch):
                        f,_,tokens=window(minted[int(rng.integers(len(minted)))])
                        key=mx.random.key(int(rng.integers(2**32)))
                        def ce(p):return loss.regularize({'head':p},f,tokens,key)*settings.regularizer_weight/settings.regularizer_batch
                        value,grads=mx.value_and_grad(ce)(trainable['head']);mx.eval(value,grads)
                        ce_total+=float(value.item())
                        accum['head']=tree_map(lambda a,b:a+b,accum['head'],grads);mx.eval(accum)
                        del value,grads
                norm=mx.sqrt(sum(mx.sum(g*g) for _,g in tree_flatten(accum)))
                scale=mx.minimum(1.,1./mx.maximum(norm,1e-12))
                accum=tree_map(lambda g:g*scale,accum);mx.eval(accum,norm)
                if not math.isfinite(float(norm.item())):raise FloatingPointError('Nonfinite training gradient; previous checkpoints retained')
                mult=min(1,step/50)*(.2+.8*.5*(1+math.cos(math.pi*step/settings.steps)))
                for k in active:
                    optimizers[k].learning_rate=rates[k]*mult
                    trainable[k]=optimizers[k].apply_gradients(accum[k],trainable[k])
                mx.eval(trainable,[o.state for o in optimizers.values()])
                params={**frozen,**trainable};del accum
                row=dict(step=step,flow_loss=total,regularizer_loss=ce_total,gradient_norm=float(norm.item()),seconds=time.perf_counter()-started,
                         peak_gib=mx.get_peak_memory()/2**30)
                print(f'train: {step}/{settings.steps} · loss {total:.6f} · {row["peak_gib"]:.2f} GiB',flush=True)
                if step%settings.save_every==0 or step==settings.steps or stopping[0]:
                    metric=evaluate()
                    row['validation_loss']=metric
                history.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if 'validation_loss' in row:
                    improved=metric<best
                    best=min(best,metric)
                    if improved:best_checkpoint=str((output/'checkpoints'/f'step-{step:08d}').resolve())
                    folder=checkpoint();last_saved=step
                    if improved:(output/'best.json').write_text(json.dumps(dict(checkpoint=best_checkpoint,validation_loss=metric)))
                    print(f'checkpoint step {step} · validation {metric:.6f}',flush=True)
            if step>last_saved:
                metric=evaluate();checkpoint();last_saved=step
        (output/'result.json').write_text(json.dumps(dict(training=True,step=step,stopped=stopping[0],settings=asdict(settings),
            checkpoint=f'step-{last_saved:08d}',validation_loss=metric,seconds=time.perf_counter()-started,
            memory=[dict(stage='training',peak_gib=mx.get_peak_memory()/2**30)]),indent=2)+'\n')
    finally:
        signal.signal(signal.SIGTERM,old_handler)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',required=True);p.add_argument('--memory-gib',type=float,default=10)
    a=p.parse_args();train(a.config,a.output,a.model,memory_gib=a.memory_gib)
