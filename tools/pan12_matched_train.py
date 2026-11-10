#!/usr/bin/env python3
"""Matched-context partial fine-tuning, PAN12 train archive only; never opens official test."""
import os
os.environ.update({'PAN12_TRAIN_POS':'400','PAN12_TRAIN_NEG':'800','PAN12_CAL_POS':'100','PAN12_CAL_NEG':'500','PAN12_DEV_POS':'100','PAN12_DEV_NEG':'500'})
import sys,json,hashlib,random,time,gc
import numpy as np, torch
import torch.nn.functional as F
from torch import nn
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score,average_precision_score
from transformers import AutoModel,AutoTokenizer
from pan12_pilot_data import ROOT, rows,y,tr,cal,dev,render
runtime=os.environ.get('PHOC_RUNTIME')
if not runtime: raise SystemExit('Set PHOC_RUNTIME to pinned phocinae-server checkout')
sys.path.insert(0,runtime)
from phocinae.engine import Engine,QTYPES
from phocinae.tokenizer import GemmaTokenizer
MODEL='google/mobilebert-uncased'; REV='1f90a6c24c7879273a291d34a849033eba2dbc0f'
P_MODEL='Phocinae/Phocinae-Largha-150M-v1'; P_REV='a57030322df13c8c55ab7a99b8b7b176e2c5aa85'
pdir=snapshot_download(P_MODEL,revision=P_REV,local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
torch.set_num_threads(4);torch.manual_seed(42)
engine=Engine.__new__(Engine)
engine.tok=GemmaTokenizer(pdir+'/tokenizer');engine._head_cache={};engine._opt_cache={}
tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,local_files_only=True)
question={'id':'risk','type':'noul','instructions':'Does this conversation contain grooming of a minor?'}
spec=engine._question_spec(question); room=128-len(engine._build_sequence(spec,[],[0,1],64,128)[0])
assert room>0
# The exact same raw prefix is fed to both models; suffix truncation is verified for both tokenizers.
def common_prefix(text):
    text=text.replace(engine.tok.mask_token,' ')
    def fits(x): return len(engine.tok.encode(x))<=room and len(tok(x,add_special_tokens=True)['input_ids'])<=128
    if fits(text):return text,False
    lo=0;hi=len(text)
    while lo<hi:
        mid=(lo+hi+1)//2
        if fits(text[:mid]):lo=mid
        else:hi=mid-1
    prefix=text[:lo]
    while prefix and not fits(prefix):prefix=prefix[:-1]
    return prefix,True
all_ids=list(tr)+list(cal)+list(dev); texts={}; truncated={}
for i in all_ids:
    texts[i],truncated[i]=common_prefix(render(rows[i][1],10))
print('partitions',len(tr),len(cal),len(dev),'room',room,'truncated dev',sum(truncated[i] for i in dev),flush=True)
class Mobile(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=AutoModel.from_pretrained(MODEL,revision=REV,local_files_only=True);self.head=nn.Linear(self.encoder.config.hidden_size,2)
    def forward(self,b):
        h=self.encoder(**b).pooler_output
        return self.head(F.layer_norm(h,(self.encoder.config.hidden_size,)))
def configure(name,model):
    for p in model.parameters():p.requires_grad=False
    if name=='mobilebert':
        for layer in list(model.encoder.encoder.layer)[-2:]:
            for p in layer.parameters():p.requires_grad=True
        for p in model.head.parameters():p.requires_grad=True
    else:
        for layer in list(model.encoder.layers)[-2:]:
            for p in layer.parameters():p.requires_grad=True
        for module in (model.encoder.final_norm,model.head,model.scorer,model.type_emb):
            for p in module.parameters():p.requires_grad=True

def batch_input(name,indices):
    if name=='mobilebert':
        return tok([texts[i] for i in indices],padding=True,truncation=True,max_length=128,return_tensors='pt')
    items=[]
    for i in indices:
        state=engine.tok.encode(texts[i]);ids,markers=engine._build_sequence(spec,state,[0,1],64,128)
        if len(markers)!=2 or len(state)>room:raise RuntimeError('Input truncated after matching')
        items.append({'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':question})
    return engine._collate(items)
def forward(name,m,b):
    if name=='mobilebert':return m(b)
    return m(b['input_ids'],b['attention_mask'],b['marker_pos'],b['marker_mask'],b['qtype'])[0]
def train(name,m):
    params={n:p for n,p in m.named_parameters() if p.requires_grad}
    optim=torch.optim.AdamW([{'params':[p for n,p in params.items() if n.startswith('encoder.')],'lr':2e-5},
                            {'params':[p for n,p in params.items() if not n.startswith('encoder.')],'lr':1e-4}],weight_decay=0.01)
    weights=torch.tensor([1.0,float((len(tr)-sum(y[tr]))/sum(y[tr]))])
    rng=random.Random(42);order=list(tr); epochs=[]
    for epoch in range(2):
        rng.shuffle(order);m.train();m.encoder.eval()
        layers=list(m.encoder.encoder.layer) if name=='mobilebert' else list(m.encoder.layers)
        for layer in layers[-2:]:layer.train()
        start=time.perf_counter(); total=0.0
        for off in range(0,len(order),4):
            ids=order[off:off+4];b=batch_input(name,ids); target=torch.tensor(y[ids],dtype=torch.long)
            optim.zero_grad(set_to_none=True); loss=F.cross_entropy(forward(name,m,b),target,weight=weights)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(list(params.values()),1.0);optim.step();total+=loss.item()
        e={'epoch':epoch+1,'loss':total/((len(order)+3)//4),'seconds':time.perf_counter()-start};epochs.append(e);print(name,e,flush=True)
    return params,epochs
def scores(name,m,indices):
    m.eval();out=[]
    with torch.inference_mode():
        for off in range(0,len(indices),8):
            logits=forward(name,m,batch_input(name,indices[off:off+8]));out.extend(torch.softmax(logits.float(),dim=1)[:,1].tolist())
    return np.array(out)
def choose_threshold(p,labels):
    for v in sorted(set(p),reverse=True):
        if recall_score(labels,p>=v,zero_division=0)>=.9:return float(v)
    return 0.0
def stats(prob,labels,t):
    pred=prob>=t;tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return {'n':len(labels),'positives':int(sum(labels)),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
            'precision':precision_score(labels,pred,zero_division=0),'recall':recall_score(labels,pred,zero_division=0),
            'f1':f1_score(labels,pred,zero_division=0),'average_precision':average_precision_score(labels,prob)}
results={}; folder=ROOT/'artifacts/pan12-matched';folder.mkdir(parents=True,exist_ok=True)
for name in ('mobilebert','phocinae'):
    torch.manual_seed(42)
    if name=='mobilebert': model=Mobile()
    else:
        engine=Engine(pdir,device='cpu',warm=False,threads=4)
        model=engine.model
    configure(name,model)
    params,epochs=train(name,model)
    t=choose_threshold(scores(name,model,cal),y[cal]);metrics=stats(scores(name,model,dev),y[dev],t)
    dest=folder/(name+'.safetensors');save_file({n:p.detach().cpu().contiguous() for n,p in params.items()},str(dest))
    results[name]={'threshold':t,'development':metrics,'epochs':epochs,'trainable_parameters':sum(p.numel() for p in params.values()),
                   'artifact':str(dest.relative_to(ROOT)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    print(name,'development',metrics,flush=True)
    del params,model
    if name=='phocinae': del engine
    gc.collect()
# Selection policy is defined in advance: best F1 among models meeting 0.85 dev recall, otherwise highest recall.
eligible=[n for n in results if results[n]['development']['recall']>=.85]
winner=max(eligible,key=lambda n:(results[n]['development']['f1'],n)) if eligible else max(results,key=lambda n:(results[n]['development']['recall'],n))
report={'protocol':'same raw text prefix fitting both tokenizers, maximum 10 messages and 128 model tokens; exactly shared sampled groups; both partially fine-tuned for 2 epochs',
 'seed':42,'context_budget_phocinae':room,'development_truncated':int(sum(truncated[i] for i in dev)),
 'partitions':{k:{'n':len(v),'positive':int(sum(y[v]))} for k,v in [('train',tr),('calibration',cal),('development',dev)]},
 'threshold_policy':'highest sampled calibration score with recall >=0.9; no probability calibration',
 'selection_policy':'highest development F1 among models with recall >=0.85, else highest recall; tie by model name',
 'models':results,'selected':winner,'selection_locked_before_test':True,
 'limitations':'Sampled pilot; author-derived weak segment labels, source cues, incomplete approximate deduplication, prompt overhead, different model heads and tokenizers, pretraining overlap unknown; no official test read in this script.'}
(ROOT/'reports/pan12-matched-selection.json').write_text(json.dumps(report,indent=2)+'\n');print('locked selection',winner,flush=True)
