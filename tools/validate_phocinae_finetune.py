#!/usr/bin/env python3
"""Restore the saved fine-tuned parameters and verify held-out aggregate counts."""
import hashlib,json,os,sys
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix
from pan12_pilot_data import ROOT,rows,y,dev,render
runtime=os.environ.get('PHOC_RUNTIME')
if not runtime: raise SystemExit('Set PHOC_RUNTIME to pinned phocinae-server checkout')
sys.path.insert(0,runtime)
from phocinae.engine import Engine,QTYPES
repo='Phocinae/Phocinae-Largha-150M-v1'; revision='a57030322df13c8c55ab7a99b8b7b176e2c5aa85'
path=snapshot_download(repo,revision=revision,local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
engine=Engine(path,device='cpu',warm=False,threads=4)
report=json.loads((ROOT/'reports/pan12-phocinae-finetune.json').read_text())
weights_path=ROOT/report['artifact']['path']
assert hashlib.sha256(weights_path.read_bytes()).hexdigest()==report['artifact']['sha256']
weights=load_file(str(weights_path)); result=engine.model.load_state_dict(weights,strict=False)
assert not result.unexpected_keys and len(weights)>0
assert set(weights).isdisjoint(result.missing_keys)
question={'id':'risk','type':'noul','instructions':'Does this conversation contain grooming of a minor?'}
scores=[]
for off in range(0,len(dev),4):
    items=[]
    for i in dev[off:off+4]:
        context=render(rows[i][1],10); spec=engine._question_spec(question)
        ids,markers=engine._build_sequence(spec,engine.tok.encode(context.replace(engine.tok.mask_token,' ')),[0,1],64,128)
        items.append({'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':question})
    logits,act=engine._forward(engine._collate(items))
    scores.extend(v[3][1] for v in engine._decode_rows(logits.numpy(),act.numpy(),items))
pred=np.asarray(scores)>=report['threshold']
tn,fp,fn,tp=confusion_matrix(y[dev],pred,labels=[0,1]).ravel()
expected=report['development']['10']
assert (tp,fp,tn,fn)==(expected['tp'],expected['fp'],expected['tn'],expected['fn'])
print('Saved parameter integrity and restored development confusion counts verified.')
