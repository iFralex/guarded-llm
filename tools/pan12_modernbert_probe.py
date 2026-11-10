#!/usr/bin/env python3
"""Resource-bounded ModernBERT frozen-encoder probe on shared PAN12 pilot split."""
import json, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from pan12_pilot_data import ROOT, rows, y, tr, cal, dev, render

MODEL='answerdotai/ModernBERT-base'
# Set this to the exact inspected snapshot before running; overrides accepted only for replication.
import os
revision=os.environ.get('MODERNBERT_REVISION')
if not revision: raise SystemExit('Set MODERNBERT_REVISION to the immutable model revision')
tok=AutoTokenizer.from_pretrained(MODEL,revision=revision,local_files_only=True)
model=AutoModel.from_pretrained(MODEL,revision=revision,local_files_only=True,attn_implementation='eager').eval()
torch.set_num_threads(4)
def features(indices,window):
    result=[]; start=time.perf_counter()
    for off in range(0,len(indices),4):
        batch=tok([render(rows[i][1],window) for i in indices[off:off+4]],padding=True,truncation=True,max_length=128,return_tensors='pt')
        with torch.inference_mode():
            hidden=model(**batch).last_hidden_state[:,0,:]
            result.append(torch.nn.functional.layer_norm(hidden,(hidden.shape[-1],)).numpy())
        if off and off%100==0: print('window',window,'progress',off,'/',len(indices),flush=True)
    print('window',window,'seconds',round(time.perf_counter()-start,2),flush=True)
    return np.concatenate(result)
head=LogisticRegression(class_weight='balanced',max_iter=500,random_state=42,solver='liblinear')
head.fit(features(tr,10),y[tr])
calibration=head.predict_proba(features(cal,10))[:,1]
cut=0.0
for v in sorted(set(calibration),reverse=True):
    if recall_score(y[cal],calibration>=v,zero_division=0)>=.9: cut=float(v); break
results={}
for window in (5,10,20):
    scores=head.predict_proba(features(dev,window))[:,1]; pred=scores>=cut
    tn,fp,fn,tp=confusion_matrix(y[dev],pred,labels=[0,1]).ravel()
    results[str(window)]={'n':len(dev),'positives':int(sum(y[dev])),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
        'precision':precision_score(y[dev],pred,zero_division=0),'recall':recall_score(y[dev],pred,zero_division=0),
        'f1':f1_score(y[dev],pred,zero_division=0),'average_precision':average_precision_score(y[dev],scores)}
report={'model':MODEL,'revision':revision,'method':'frozen encoder, layer-normalized CLS features and balanced logistic head; no encoder fine-tuning',
        'protocol':'identical sampled author/duplicate-disjoint partitions and 5/10/20 windows as MobileBERT pilot',
        'threshold_policy':'highest calibration score with >=0.9 sampled calibration recall; no probability calibration',
        'threshold':cut,'development':results,'limitations':'Small sampled pilot, weak author labels, source bias, limited near-duplicate check, unknown pretraining overlap; no official test or Android test.'}
(ROOT/'reports/pan12-modernbert-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
