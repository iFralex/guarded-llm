#!/usr/bin/env python3
"""Offline zero-shot Laya test on the shared PAN12 pilot; no chat text is saved."""
import json,os,sys,time
import numpy as np
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
from pan12_pilot_data import ROOT,rows,y,cal,dev,render

runtime=os.environ.get('LAYA_RUNTIME')
if not runtime: raise SystemExit('Set LAYA_RUNTIME to the pinned Laya source checkout')
sys.path.insert(0,runtime)
import laya
repo='convaiinnovations/laya'; revision='7b928d828b7b0e022f929d9bd2e44165aa270148'
path=snapshot_download(repo,revision=revision,local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/*','tokenizer/*'])
torch.set_num_threads(4)
agent=laya.load(path,device='cpu')
questions={'risk':{'type':'noul','instructions':'Does this conversation contain grooming of a minor?'}}
timings=[]
def predict(indices,window):
    scores=[]; start=time.perf_counter()
    for off in range(0,len(indices),4):
        contexts=[render(rows[i][1],window) for i in indices[off:off+4]]
        results=agent.predict_batch(contexts,questions,batch_size=4,max_len=128,head_max_len=64)
        scores.extend(float(r['answers']['risk']['noul']) for r in results)
        if off and off%100==0: print('window',window,'progress',off,'/',len(indices),flush=True)
    elapsed=time.perf_counter()-start
    timings.append({'window':window,'samples':len(indices),'seconds':elapsed,'batch_size':4})
    print('window',window,'seconds',round(elapsed,2),flush=True)
    return np.asarray(scores)
fixture='Hello. How is your day?'
a=agent.system_one(fixture,questions,max_len=128,head_max_len=64)['answers']['risk']['noul']
b=agent.predict_batch([fixture],questions,batch_size=1,max_len=128,head_max_len=64)[0]['answers']['risk']['noul']
if abs(float(a)-float(b))>1e-4: raise RuntimeError('Batched inference does not match single-input API')
print('short-input batching parity verified',flush=True)
calibration=predict(cal,10); cut=0.0
for v in sorted(set(calibration),reverse=True):
    if recall_score(y[cal],calibration>=v,zero_division=0)>=.9: cut=float(v);break
out={}
for n in (5,10,20):
    score=predict(dev,n); pred=score>=cut
    tn,fp,fn,tp=confusion_matrix(y[dev],pred,labels=[0,1]).ravel()
    out[str(n)]={'n':len(dev),'positives':int(sum(y[dev])),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
       'precision':precision_score(y[dev],pred,zero_division=0),'recall':recall_score(y[dev],pred,zero_division=0),
       'f1':f1_score(y[dev],pred,zero_division=0),'average_precision':average_precision_score(y[dev],score)}
report={'model':repo,'revision':revision,'runtime_revision':'3c346a2f8b542838eed6a1d210aa378c79e71c4c',
 'mode':'zero-shot typed binary decision; no fine-tuning','input':'128 tokens total including instruction and options; batch/single API parity checked',
 'threshold_policy':'highest score achieving >=0.9 recall on sampled calibration; not PAN-specific probability calibration',
 'threshold':cut,'development':out,'host_batch_timings':timings,
 'limitations':'Prompt-dependent zero-shot probe; weak author labels, sampled data, source/decoy bias, partial near-deduplication, unknown pretraining overlap; no Android evaluation or official PAN12 test.'}
(ROOT/'reports/pan12-laya-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
