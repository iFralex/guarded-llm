#!/usr/bin/env python3
"""Small, author-disjoint PAN12 pilot: TF-IDF vs frozen MobileBERT + fitted head.

Only the official training archive is read. No examples, identifiers or checkpoints
containing training text are written to the report. Resource-bounded pilot, not a final test.
"""
import hashlib, io, json, os, random, re, time, zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

from pan12_pilot_data import ROOT, ARCHIVE, DIGEST, rows, y, tr, cal, dev, render, drop_cal, drop_dev
MODEL = 'google/mobilebert-uncased'
REVISION = '1f90a6c24c7879273a291d34a849033eba2dbc0f'

# Threshold chosen on calibration to achieve at least 90% recall, if possible.
def threshold(probs, labels):
    candidates=sorted(set(probs),reverse=True)
    for value in candidates:
        if recall_score(labels,np.array(probs)>=value,zero_division=0)>=.9: return float(value)
    return 0.0
def metrics(labels, probs, limit):
    pred=np.array(probs)>=limit; tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return {'n':len(labels),'positives':int(sum(labels)), 'precision':precision_score(labels,pred,zero_division=0),
        'recall':recall_score(labels,pred,zero_division=0),'f1':f1_score(labels,pred,zero_division=0),
        'average_precision':average_precision_score(labels,probs),'brier_uncalibrated':brier_score_loss(labels,probs),
        'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'false_alerts_per_100_negative_windows':float(100*fp/max(1,fp+tn))}
text=lambda ids,n:[render(rows[i][1],n) for i in ids]
# Training on the same 10-message inputs for both systems; evaluation varies the available prefix.
ytrain=y[tr]; ycal=y[cal]; ydev=y[dev]
tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
# Give TF-IDF the same MobileBERT-tokenized and truncated text budget.
def baseline_text(ids,n):
    tokens=tokenizer(text(ids,n),truncation=True,max_length=128)['input_ids']
    return tokenizer.batch_decode(tokens,skip_special_tokens=True)
vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True,dtype=np.float32)
lin=LogisticRegression(class_weight='balanced',max_iter=300,random_state=42,solver='liblinear')
lin.fit(vec.fit_transform(baseline_text(tr,10)),ytrain)
base_cal=lin.predict_proba(vec.transform(baseline_text(cal,10)))[:,1]; base_threshold=threshold(base_cal,ycal)
base={str(n):metrics(ydev,lin.predict_proba(vec.transform(baseline_text(dev,n)))[:,1],base_threshold) for n in (5,10,20)}
print('TF-IDF baseline finished',flush=True)

encoder=AutoModel.from_pretrained(MODEL,revision=REVISION,local_files_only=True).eval()
torch.set_num_threads(min(4,torch.get_num_threads()))
def embed(texts,tag):
    result=[]; start=time.perf_counter()
    for start_i in range(0,len(texts),8):
        data=tokenizer(texts[start_i:start_i+8],padding=True,truncation=True,max_length=128,return_tensors='pt')
        with torch.inference_mode(): result.append(torch.nn.functional.layer_norm(encoder(**data).pooler_output, (encoder.config.hidden_size,)).cpu().numpy())
        if start_i and start_i%160==0: print(tag,start_i,'/',len(texts),flush=True)
    print(tag,'seconds',round(time.perf_counter()-start,2),flush=True)
    return np.concatenate(result)
Xtrain=embed(text(tr,10),'train')
head=LogisticRegression(class_weight='balanced',max_iter=500,random_state=42,solver='liblinear')
head.fit(Xtrain,ytrain)
Xcal=embed(text(cal,10),'calibration')
cal_probs=head.predict_proba(Xcal)[:,1]; selected=threshold(cal_probs,ycal)
mobile={}; dev_features={}
for n in (5,10,20):
    dev_features[n]=embed(text(dev,n),f'development-{n}')
    mobile[str(n)]=metrics(ydev,head.predict_proba(dev_features[n])[:,1],selected)
# Turn-count proxy only, not actual timestamps or a validated grooming onset.
pos_positions=[i for i,v in enumerate(ydev) if v==1]
alarms={n:head.predict_proba(dev_features[n])[:,1]>=selected for n in (5,10,20)}
early=Counter(str(next((n for n in (5,10,20) if alarms[n][i]),'none')) for i in pos_positions)

# Export frozen encoder plus the fitted head to ONNX. This is a sequence classifier,
# not integration into Android; only model weights and aggregate metrics are saved.
class Classifier(torch.nn.Module):
    def __init__(self,model,coef,bias):
        super().__init__(); self.encoder=model; self.head=torch.nn.Linear(coef.shape[1],1)
        with torch.no_grad():
            self.head.weight.copy_(torch.from_numpy(coef.astype('float32')))
            self.head.bias.copy_(torch.tensor(bias,dtype=torch.float32))
    def forward(self,input_ids,attention_mask,token_type_ids):
        return self.head(torch.nn.functional.layer_norm(self.encoder(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).pooler_output, (self.encoder.config.hidden_size,)))
model=Classifier(encoder,head.coef_,head.intercept_)
# A newly initialized head avoids instability from the frozen probe's large coefficients.
torch.manual_seed(42)
model.head.reset_parameters()
# Fine-tune all encoder weights and the head on the same small training cohort.
# Full PAN12 fine-tuning and hyperparameter selection remain future work.
optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':2e-5},{'params':model.head.parameters(),'lr':1e-3}],weight_decay=0.01)
loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(ytrain)-sum(ytrain))/sum(ytrain)],dtype=torch.float32))
training_text=text(tr,10); order=list(range(len(tr))); train_rng=random.Random(42)
torch.manual_seed(42)
for epoch in range(2):
    train_rng.shuffle(order); model.train(); loss_sum=0.0; start=time.perf_counter()
    for offset in range(0,len(order),8):
        batch_indices=order[offset:offset+8]
        inputs=tokenizer([training_text[i] for i in batch_indices],padding=True,truncation=True,max_length=128,return_tensors='pt')
        target=torch.tensor(ytrain[batch_indices],dtype=torch.float32).reshape(-1,1)
        optimizer.zero_grad(set_to_none=True)
        logits=model(inputs['input_ids'],inputs['attention_mask'],inputs['token_type_ids'])
        loss=loss_fn(logits,target); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); loss_sum+=float(loss.detach())
    print('fine-tune epoch',epoch+1,'loss',round(loss_sum/((len(order)+7)//8),4),'seconds',round(time.perf_counter()-start,2),flush=True)
model.eval()
def finetuned_predict(examples):
    scores=[]
    for offset in range(0,len(examples),8):
        batch=tokenizer(examples[offset:offset+8],padding=True,truncation=True,max_length=128,return_tensors='pt')
        with torch.inference_mode(): scores.extend(torch.sigmoid(model(batch['input_ids'],batch['attention_mask'],batch['token_type_ids'])).reshape(-1).tolist())
    return np.array(scores)
ft_cal=finetuned_predict(text(cal,10)); ft_threshold=threshold(ft_cal,ycal)
ft_dev={n:finetuned_predict(text(dev,n)) for n in (5,10,20)}
ft_metrics={str(n):metrics(ydev,ft_dev[n],ft_threshold) for n in (5,10,20)}
ft_alarms={n:ft_dev[n]>=ft_threshold for n in (5,10,20)}
ft_early=Counter(str(next((n for n in (5,10,20) if ft_alarms[n][i]),'none')) for i in pos_positions)
out=ROOT/'artifacts/pan12-mobilebert-pilot'; out.mkdir(parents=True,exist_ok=True)
tokenizer.save_pretrained(out)
dummy=tokenizer(['Example conversation for export.'],return_tensors='pt')
path=out/'model.onnx'
torch.onnx.export(model,(dummy['input_ids'],dummy['attention_mask'],dummy['token_type_ids']),str(path),
    input_names=['input_ids','attention_mask','token_type_ids'],output_names=['logits'],opset_version=17,
    dynamic_axes={k:{0:'batch',1:'sequence'} for k in ['input_ids','attention_mask','token_type_ids']}|{'logits':{0:'batch'}})
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
quant=out/'model-int8.onnx'
quantize_dynamic(str(path),str(quant),weight_type=QuantType.QInt8)
quant_u8=out/'model-uint8.onnx'
quantize_dynamic(str(path),str(quant_u8),weight_type=QuantType.QUInt8)
def run_onnx(p,examples):
    session=ort.InferenceSession(str(p),providers=['CPUExecutionProvider']); names={i.name for i in session.get_inputs()}; values=[]
    for i in range(0,len(examples),8):
        batch=tokenizer(examples[i:i+8],padding=True,truncation=True,max_length=128,return_tensors='np')
        logits=session.run(None,{k:v.astype(np.int64) for k,v in batch.items() if k in names})[0].reshape(-1)
        values.extend((1/(1+np.exp(-logits))).tolist())
    return np.array(values)
samples=text(dev,10); fp=run_onnx(path,samples); q=run_onnx(quant,samples); q8=run_onnx(quant_u8,samples)
py=ft_dev[10]
report={'archive_sha256':DIGEST,'model':MODEL,'revision':REVISION,'protocol':'pilot: first 5/10/20 messages; model fitted on 10; same author/text-disjoint sampled cohorts; TF-IDF sees MobileBERT-truncated decoded input',
 'sampling':'train <=120 positive, <=240 negative; calibration/development each <=35 positive, <=175 negative; not natural prevalence',
 'partitions':{'train':{'n':len(tr),'positive':int(sum(ytrain))},'calibration':{'n':len(cal),'positive':int(sum(ycal))},'development':{'n':len(dev),'positive':int(sum(ydev))}},
 'near_duplicate_filter':'3-word shingles Jaccard >=0.8 between selected 10-message windows; texts with <8 shingles not checked; near-duplicate audit incomplete',
 'near_duplicates_dropped':{'calibration':drop_cal,'development':drop_dev},
 'threshold_policy':'highest calibration score at which sampled calibration recall >=0.9; no probability calibration',
 'tfidf':{'threshold':base_threshold,'development':base},
 'mobilebert_frozen_encoder_logistic_head':{'threshold':selected,'development':mobile,'positive_first_alarm_prefix':dict(early)},
 'mobilebert_finetuned':{'threshold':ft_threshold,'epochs':2,'encoder_learning_rate':2e-5,'head_learning_rate':1e-3,'development':ft_metrics,'positive_first_alarm_prefix':dict(ft_early)},
 'onnx':{'bytes':path.stat().st_size,'int8_bytes':quant.stat().st_size,'max_probability_error_float_vs_python':float(np.max(np.abs(fp-py))),
         'max_probability_error_int8_vs_float':float(np.max(np.abs(fp-q))), 'int8_development_10':metrics(ydev,q,ft_threshold),
         'uint8_bytes':quant_u8.stat().st_size, 'max_probability_error_uint8_vs_float':float(np.max(np.abs(fp-q8))), 'uint8_development_10':metrics(ydev,q8,ft_threshold),
         'float_development_10':metrics(ydev,fp,ft_threshold)},
 'limitations':'Small sampled pilot with altered prevalence; author-derived labels, source/decoy cues, approximate deduplication, unknown model-pretraining overlap; no PAN12 official test, real-time or Android measurements; false alarms are per negative window, not per user or day.'}
(ROOT/'reports').mkdir(exist_ok=True)
(ROOT/'reports/pan12-model-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
