#!/usr/bin/env python3
"""Matched-prefix research comparison; no conversation texts leave memory."""
import argparse, gc, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from cyby24_split import ROOT, load

parser=argparse.ArgumentParser(); parser.add_argument('--model',choices=['mobilebert','modernbert','phocinae','roberta'],required=True); args=parser.parse_args()
name=args.model; out=ROOT/f'reports/cyby24-matched-{name}.json'
if out.exists(): raise SystemExit('Report already exists; refusing to overwrite exploratory evaluation')
runtime=os.environ.get('PHOC_RUNTIME');
if not runtime: raise SystemExit('Set PHOC_RUNTIME to pinned phocinae-server')
sys.path.insert(0,runtime)
from phocinae.engine import Engine, QTYPES
from phocinae.tokenizer import GemmaTokenizer
MOBILE='google/mobilebert-uncased'; MOBILE_REV='1f90a6c24c7879273a291d34a849033eba2dbc0f'
PHOC='Phocinae/Phocinae-Largha-150M-v1'; PHOC_REV='a57030322df13c8c55ab7a99b8b7b176e2c5aa85'
PDIR=snapshot_download(PHOC,revision=PHOC_REV,local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
MODERN=ROOT/'.cache/modernbert-base'; ROBERTA=ROOT/'.cache/roberta-cyberbullying'
torch.set_num_threads(4);torch.manual_seed(79); random.seed(79);np.random.seed(79)
conversations,y,tr,dev,test,components=load()
mobile_tok=AutoTokenizer.from_pretrained(MOBILE,revision=MOBILE_REV,local_files_only=True)
modern_tok=AutoTokenizer.from_pretrained(MODERN,local_files_only=True)
roberta_tok=AutoTokenizer.from_pretrained(ROBERTA,local_files_only=True)
engine=Engine.__new__(Engine);engine.tok=GemmaTokenizer(PDIR+'/tokenizer');engine._head_cache={};engine._opt_cache={}
question={'id':'risk','type':'noul','instructions':'Does this conversation contain cyberbullying?'}
spec=engine._question_spec(question);budget=256;room=budget-len(engine._build_sequence(spec,[],[0,1],64,budget)[0]); assert room>0
def prefix(text):
    text=text.replace(engine.tok.mask_token,' ')
    def fits(s):return (len(engine.tok.encode(s))<=room and all(len(t(s,add_special_tokens=True)['input_ids'])<=budget for t in (mobile_tok,modern_tok,roberta_tok)))
    if fits(text):return text,False
    lo,hi=0,len(text)
    while lo<hi:
        mid=(lo+hi+1)//2
        if fits(text[:mid]):lo=mid
        else:hi=mid-1
    while lo and not fits(text[:lo]):lo-=1
    return text[:lo],True
texts={};shortened={}
for i in list(tr)+list(dev)+list(test):
    original=' '.join(f'Message {j+1}: {s}' for j,s in enumerate(conversations[i]))
    texts[i],shortened[i]=prefix(original)
print('prepared shared inputs',len(texts),'phocinae text budget',room,'test truncated',sum(shortened[i] for i in test),flush=True)
class EncoderClassifier(nn.Module):
    def __init__(self,which):
        super().__init__(); self.which=which
        if which=='mobilebert': self.encoder=AutoModel.from_pretrained(MOBILE,revision=MOBILE_REV,local_files_only=True)
        else:self.encoder=AutoModel.from_pretrained(MODERN,local_files_only=True,attn_implementation='eager')
        self.head=nn.Linear(self.encoder.config.hidden_size,2)
    def forward(self,b):
        h=self.encoder(**b)
        z=h.pooler_output if self.which=='mobilebert' else h.last_hidden_state[:,0,:]
        return self.head(F.layer_norm(z,(self.encoder.config.hidden_size,)))
def verify_weights(path,expected):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    if h.hexdigest()!=expected:raise RuntimeError('Model file checksum mismatch')
if name=='roberta':
    verify_weights(ROBERTA/'model.safetensors','9cf3d24e3d64cbad3778219fb24a3e0390a523b50b878f41a99ca0b3656e57df')
    model=AutoModelForSequenceClassification.from_pretrained(ROBERTA,local_files_only=True);tokenizer=roberta_tok
elif name in ('mobilebert','modernbert'):
    if name=='modernbert':verify_weights(MODERN/'model.safetensors','340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9')
    model=EncoderClassifier(name);tokenizer=mobile_tok if name=='mobilebert' else modern_tok
else:
    engine=Engine(PDIR,device='cpu',warm=False,threads=4);model=engine.model
def batch(indices):
    if name!='phocinae':return tokenizer([texts[i] for i in indices],padding=True,truncation=True,max_length=budget,return_tensors='pt')
    items=[]
    for i in indices:
        state=engine.tok.encode(texts[i]);ids,markers=engine._build_sequence(spec,state,[0,1],64,budget)
        assert len(state)<=room and len(markers)==2
        items.append({'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':question})
    return engine._collate(items)
def forward(b):
    if name=='roberta':return model(**b).logits
    if name!='phocinae':return model(b)
    return model(b['input_ids'],b['attention_mask'],b['marker_pos'],b['marker_mask'],b['qtype'])[0]
epochs=[];artifact=None;trainable_count=0
if name!='roberta':
    for p in model.parameters():p.requires_grad=False
    if name=='mobilebert':layers=list(model.encoder.encoder.layer)
    elif name=='modernbert':layers=list(model.encoder.layers)
    else:layers=list(model.encoder.layers)
    for layer in layers[-2:]:
        for p in layer.parameters():p.requires_grad=True
    if name=='phocinae':
        for module in (model.encoder.final_norm,model.head,model.scorer,model.type_emb):
            for p in module.parameters():p.requires_grad=True
    else:
        for p in model.head.parameters():p.requires_grad=True
    params={n:p for n,p in model.named_parameters() if p.requires_grad};trainable_count=sum(p.numel() for p in params.values())
    optim=torch.optim.AdamW([{'params':[p for n,p in params.items() if n.startswith('encoder.')],'lr':2e-5}, {'params':[p for n,p in params.items() if not n.startswith('encoder.')],'lr':1e-4}],weight_decay=.01)
    class_weight=torch.tensor([1.,float((len(tr)-sum(y[tr]))/sum(y[tr]))]);rng=random.Random(79); order=list(tr)
    for epoch in range(2):
        rng.shuffle(order);model.train();model.encoder.eval()
        for layer in layers[-2:]:layer.train()
        begin=time.perf_counter();loss_sum=0.
        for off in range(0,len(order),2):
            ids=order[off:off+2]; optim.zero_grad(set_to_none=True)
            loss=F.cross_entropy(forward(batch(ids)),torch.tensor(y[ids],dtype=torch.long),weight=class_weight)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(list(params.values()),1.);optim.step();loss_sum+=loss.item()
        e={'epoch':epoch+1,'loss':loss_sum/((len(order)+1)//2),'seconds':time.perf_counter()-begin};epochs.append(e);print('epoch',e,flush=True)
    folder=ROOT/'artifacts/cyby24-matched';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.safetensors');save_file({n:p.detach().cpu().contiguous() for n,p in params.items()},str(path));artifact={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def probabilities(indices):
    model.eval();res=[]
    with torch.inference_mode():
        for off in range(0,len(indices),4):res.extend(torch.softmax(forward(batch(indices[off:off+4])).float(),dim=1)[:,1].tolist())
    return np.asarray(res)
cal=probabilities(dev); threshold_candidates=[]
for threshold in sorted(set(cal),reverse=True):
    pred=cal>=threshold; r=recall_score(y[dev],pred,zero_division=0)
    if r>=.8:threshold_candidates.append((precision_score(y[dev],pred,zero_division=0),float(threshold)))
if not threshold_candidates:raise RuntimeError('No threshold achieves calibration recall')
threshold=max(threshold_candidates)[1]
def stats(indices,probs):
    labels=y[indices];pred=probs>=threshold;tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return {'n':len(labels),'positive':int(sum(labels)),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'precision':precision_score(labels,pred,zero_division=0),'recall':recall_score(labels,pred,zero_division=0),'f1':f1_score(labels,pred,zero_division=0),'average_precision':average_precision_score(labels,probs)}
development=stats(dev,cal);print('development',development,'threshold',threshold,flush=True)
test_scores=probabilities(test);evaluation=stats(test,test_scores)
report={'model':name,'seed':79,'data':'CYBY24: thread labels 3/4 positive, 1/2 negative; prior public source used for exploration','protocol':'same raw thread prefix fits all four tokenizers and Phocinae question, 256 total tokens per model; grouped by exact message and tweet ID; 2 epochs, final two layers and head for MobileBERT/ModernBERT/Phocinae; pre-fine-tuned RoBERTa not retrained',
 'partitions':{k:{'n':len(ix),'positive':int(sum(y[ix])),'truncated':int(sum(shortened[i] for i in ix))} for k,ix in [('train',tr),('calibration',dev),('test',test)]},
 'phocinae_context_tokens':room,'threshold_policy':'on calibration, maximum precision among score thresholds achieving recall >=0.8; score is not a calibrated probability', 'threshold':threshold,'development':development,'test':evaluation,'epochs':epochs,'trainable_parameters':trainable_count,'artifact':artifact,
 'limitations':'Exploratory previously inspected same-source test, not independent or Instagram; thread labels weak for message-level inference; class prevalence source-specific; cross-author and near-duplicate leakage possible; same raw prefixes but tokenization and model heads differ; pretraining and source overlap unknown; no Android measurements.'}
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
