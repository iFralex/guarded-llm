#!/usr/bin/env python3
"""Compare quantization variants on the pilot calibration/development split."""
import json
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import confusion_matrix, precision_score, recall_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer
from pan12_pilot_data import ROOT, rows, y, cal, dev, render

base=ROOT/'artifacts/pan12-mobilebert-pilot'
fp=base/'model.onnx'; pc=base/'model-int8-per-channel.onnx'
quantize_dynamic(str(fp),str(pc),weight_type=QuantType.QInt8,per_channel=True)
tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
def predict(path,indices):
    opts=ort.SessionOptions(); opts.intra_op_num_threads=4; opts.inter_op_num_threads=1
    s=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider']); names={x.name for x in s.get_inputs()}; outputs=[]
    for i in range(0,len(indices),8):
        texts=[render(rows[idx][1],10) for idx in indices[i:i+8]]
        batch=tok(texts,return_tensors='np',padding=True,truncation=True,max_length=128)
        logits=s.run(None,{k:v.astype(np.int64) for k,v in batch.items() if k in names})[0].reshape(-1)
        outputs.extend((1/(1+np.exp(-logits))).tolist())
    return np.array(outputs)
def select(scores,labels):
    for value in sorted(set(scores),reverse=True):
        if recall_score(labels,scores>=value,zero_division=0)>=.9: return float(value)
    return 0.0
def summarize(scores,labels,threshold):
    pred=scores>=threshold; tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return {'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn), 'precision':precision_score(labels,pred,zero_division=0),
            'recall':recall_score(labels,pred,zero_division=0),'average_precision':average_precision_score(labels,scores)}
def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)
def ece(probs,labels):
    error=0.0
    for k in range(10):
        mask=(probs>=k/10)&(probs<((k+1)/10) if k<9 else probs<=1)
        if mask.any(): error+=float(mask.mean())*abs(float(probs[mask].mean())-float(np.asarray(labels)[mask].mean()))
    return error
ref=predict(fp,dev); result={}
for name in ['model.onnx','model-int8.onnx','model-uint8.onnx','model-int8-per-channel.onnx']:
    p=base/name; calibration=predict(p,cal); scores=predict(p,dev)
    cut=select(calibration,y[cal])
    scaler=LogisticRegression(random_state=42).fit(logit(calibration),y[cal])
    calibrated=scaler.predict_proba(logit(scores))[:,1]
    result[name]={'bytes':p.stat().st_size,'threshold':cut,
        'max_abs_probability_error_vs_float':float(np.max(np.abs(scores-ref))),
        'mean_abs_probability_error_vs_float':float(np.mean(np.abs(scores-ref))),
        'development':summarize(scores,y[dev],cut),
        'probability_calibration':{'method':'Platt logistic scaling on calibration only; sampled prevalence',
         'brier_raw':brier_score_loss(y[dev],scores),'brier_platt':brier_score_loss(y[dev],calibrated),
         'ece_10_bins_raw':ece(scores,y[dev]),'ece_10_bins_platt':ece(calibrated,y[dev])}}
truncation={}
for n in (5,10,20):
    lengths=[len(tok(render(rows[i][1],n),truncation=False)['input_ids']) for i in dev]
    truncation[str(n)]={'n':len(lengths),'over_128_tokens':sum(v>128 for v in lengths)}
report={'development_truncation':truncation, 'policy':'Threshold and Platt scaling fitted independently per variant using calibration only; both depend on sampled prevalence.',
        'limitations':'Pilot sample, weak labels and possible data artifacts; no Android runtime or battery test. Quantized scores should not be treated as interchangeable with float scores.',
        'models':result}
(ROOT/'reports/pan12-quantization.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
