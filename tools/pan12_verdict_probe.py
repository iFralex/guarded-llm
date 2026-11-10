#!/usr/bin/env python3
"""Zero-shot Verdict probe using public ONNX formatting on the shared PAN12 pilot."""
import json,time
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
from pan12_pilot_data import ROOT,rows,y,cal,dev,render

model='heman10x/rlcd-modernbert-151m'; revision='8af2496eb63c7fa66d7d234e1f62629380030eb4'
path=snapshot_download(model,revision=revision,local_files_only=True,allow_patterns=['model.onnx','tokenizer.json','tokenizer_config.json','config.json','calibrator.json'])
# The published tokenizer config names a newer backend; load its standard tokenizer JSON locally.
tok=PreTrainedTokenizerFast(tokenizer_file=path+'/tokenizer.json',pad_token='[PAD]',cls_token='[CLS]',sep_token='[SEP]',unk_token='[UNK]',mask_token='[MASK]')
assert tok.convert_tokens_to_ids('<<LABEL>>')==50368 and tok.convert_tokens_to_ids('<<SEP>>')==50369
opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
session=ort.InferenceSession(path+'/model.onnx',opts,providers=['CPUExecutionProvider'])
inputs={i.name for i in session.get_inputs()}
assert inputs <= {'input_ids','attention_mask'}, f'Unexpected ONNX inputs: {inputs}'
proposition='This conversation contains grooming of a minor.'
labels=['true: '+proposition,'false: not '+proposition,'insufficient evidence']
# Official GLiClass formatting; scores are not PAN12-calibrated probabilities.
def prompt(context):
    return ''.join('<<LABEL>>'+x for x in labels)+'<<SEP>>'+'Context:\n'+context+'\n\nEvaluate proposition: '+proposition
def predict(indices,window):
    results=[]; start=time.perf_counter()
    for off in range(0,len(indices),4):
        prompts=[prompt(render(rows[i][1],window)) for i in indices[off:off+4]]
        batch=tok(prompts,padding=True,truncation=True,max_length=128,return_tensors='np')
        out=session.run(None,{k:v.astype(np.int64) for k,v in batch.items() if k in inputs})[0]
        assert out.ndim==2 and out.shape[1]>=3
        logits=out[:,:3]; exp=np.exp(logits-logits.max(axis=1,keepdims=True)); probs=exp/exp.sum(axis=1,keepdims=True)
        results.extend(probs[:,0].tolist())
        if off and off%100==0: print('window',window,'progress',off,'/',len(indices),flush=True)
    print('window',window,'seconds',round(time.perf_counter()-start,2),flush=True)
    return np.array(results)
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
report={'model':model,'revision':revision,'mode':'zero-shot typed three-choice decision; no task fine-tuning',
 'input':'128-token total budget includes label descriptions, proposition and chat; not the same usable text span as other models',
 'threshold_policy':'highest score achieving >=0.9 recall on sampled calibration; no PAN-specific probability calibration',
 'threshold':cut,'development':out,
 'limitations':'Prompt-dependent zero-shot probe on same sampled groups; weak author labels, dataset source bias, unknown model pretraining overlap; no Android evaluation and no official PAN12 test. False and insufficient evidence choices are not the same as ground truth.'}
(ROOT/'reports/pan12-verdict-probe.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2),flush=True)
