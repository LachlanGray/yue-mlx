"""Opt-in one real-weight gradient/update, no job submission or saved weights.

Refuses to run if the usual studio reports busy or another inference/training
worker is visible. Synthetic data checks numerics/memory, not learning quality.
"""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


def main():
    try:
        state=json.load(urlopen('http://127.0.0.1:7860/api/state',timeout=2))
        if state.get('busy'):
            print('Skipped: studio busy');return
    except OSError:pass
    processes=subprocess.check_output(['ps','-axo','pid,command'],text=True)
    workers=[line for line in processes.splitlines() if ('-m yue2_mlx.cli' in line or '-m yue2_mlx.training' in line) and not line.lstrip().startswith(str(os.getpid())+' ')]
    if workers:
        print('Skipped: inference/training worker active');return
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_map,tree_flatten
    from yue2_mlx.pipeline import Pipeline
    from yue2_mlx.training import JointLoss,load_parameters
    pipe=Pipeline('models/yue2-8bit',memory_gib=10)
    prefix=pipe.plan(style='',lyrics='',cot='off',seed=123)['prefix']
    ar=pipe.checkpoint.load('ar',head=False);nar=pipe.checkpoint.load('nar')
    params=load_parameters('models/realaudio-v4');mx.eval(params)
    f=mx.random.normal((256,1024),key=mx.random.key(1))
    z=mx.random.normal((256,64),key=mx.random.key(2))*.1
    noise=mx.random.normal(z.shape,key=mx.random.key(3))
    objective=JointLoss(ar,nar)
    print('Real-weight 256-frame joint backward',flush=True);start=time.perf_counter()
    def loss(p):return objective.flow(p,f,z,mx.array(prefix),.5,noise,key=mx.random.key(4))
    value,grads=mx.value_and_grad(loss)(params);mx.eval(value,grads)
    norm=mx.sqrt(sum(mx.sum(g*g) for _,g in tree_flatten(grads)));mx.eval(norm)
    scale=mx.minimum(1.,1/mx.maximum(norm,1e-12));grads=tree_map(lambda g:g*scale,grads);mx.eval(grads)
    assert mx.isfinite(value).item() and mx.isfinite(norm).item()
    optimizer=optim.AdamW(learning_rate=1e-5)
    updated=optimizer.apply_gradients(grads,params);mx.eval(updated,optimizer.state)
    assert all(mx.all(mx.isfinite(v)).item() for _,v in tree_flatten(updated))
    report=dict(frames=256,loss=value.item(),gradient_norm=norm.item(),seconds=time.perf_counter()-start,
                peak_gib=mx.get_peak_memory()/2**30,trainable_values=sum(v.size for _,v in tree_flatten(params)))
    print(json.dumps(report,indent=2),flush=True)
    folder=Path('outputs/training-preflight');folder.mkdir(exist_ok=True)
    (folder/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
