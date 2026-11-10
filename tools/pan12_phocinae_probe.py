#!/usr/bin/env python3
"""Zero-shot Phocinae probe using the upstream runtime on the shared PAN12 pilot."""
import json, os, sys, time
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
from pan12_pilot_data import ROOT, rows, y, cal, dev, render

runtime=os.environ.get('PHOC_RUNTIME')
if not runtime: raise SystemExit('Set PHOC_RUNTIME to a checkout of the pinned phocinae-server revision')
sys.path.insert(0,runtime)
from phocinae.engine import Engine, QTYPES
repo='Phocinae/Phocinae-Largha-150M-v1'; revision='a57030322df13c8c55ab7a99b8b7b176e2c5aa85'
path=snapshot_download(repo,revision=revision,local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
engine=Engine(path,device='cpu',warm=False,threads=4)
question={'id':'risk','type':'noul','instructions':'Does this conversation contain grooming of a minor?'}

def batch_score(contexts):
    items=[]
    for context in contexts:
        spec=engine._question_spec(question)
        state_ids=engine.tok.encode(context.replace(engine.tok.mask_token,' '))
        ids,markers=engine._build_sequence(spec,state_ids,[0,1],64,128)
        if len(markers)!=2: raise RuntimeError('Decision options exceed token budget')
        items.append({'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':question})
    logits,act=engine._forward(engine._collate(items))
    decoded=engine._decode_rows(logits.numpy(),act.numpy(),items)
    return [row[3][1] for row in decoded]

# Check batched private helpers against the author's public API on a short synthetic input.
fixture='Hello. How is your day?'
reference=engine.run(fixture,[question],with_scores=True)[4]['risk'][1]
actual=batch_score([fixture])[0]
if abs(reference-actual)>0.001: raise RuntimeError('Runtime batching differs from public API on fixture')
print('short-input batching parity verified',flush=True)
timings=[]
def predict(indices,window):
    scores=[]; start=time.perf_counter()
    for off in range(0,len(indices),4):
        scores.extend(batch_score([render(rows[i][1],window) for i in indices[off:off+4]]))
        if off and off%100==0: print('window',window,'progress',off,'/',len(indices),flush=True)
    elapsed=time.perf_counter()-start
    timings.append({'window':window,'samples':len(indices),'seconds':elapsed,'batch_size':4})
    print('window',window,'seconds',round(elapsed,2),flush=True)
    return np.asarray(scores)
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
report={'model':repo,'revision':revision,'runtime_revision':'f751d7258e50b4cd19fe3967ae184ed895ce68b5',
 'mode':'zero-shot typed binary decision; no fine-tuning','input':'128 tokens total including instruction and options; short-input private batch/public runtime parity checked',
 'threshold_policy':'highest score achieving >=0.9 recall on sampled calibration; not PAN-specific probability calibration',
 'threshold':cut,'development':out,'host_batch_timings':timings,
 'limitations':'Prompt-dependent zero-shot probe; weak author labels, sampled data, source/decoy bias, partial near-deduplication, unknown pretraining overlap, possible private runtime API drift; no phone evaluation or official PAN12 test.'}
(ROOT/'reports/pan12-phocinae-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
