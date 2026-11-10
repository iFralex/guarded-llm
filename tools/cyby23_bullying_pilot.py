#!/usr/bin/env python3
"""Small within-source bullying pilot. Never exports tweet text or identifiers."""
import hashlib, json, random, re
from collections import Counter
from pathlib import Path
import numpy as np
import onnxruntime as ort
import openpyxl
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/external/cyby23.xlsx'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'f7164c7b3d46033bb3a73e3051db7d0693668935c07222f1de997a34df483519'
DEST = ROOT/'reports/cyby23-bullying-pilot.json'
if DEST.exists(): raise SystemExit('Pilot report already exists; refusing to overwrite evaluated test.')
MODEL = 'google/mobilebert-uncased'
REV = '1f90a6c24c7879273a291d34a849033eba2dbc0f'
seed = 67
torch.set_num_threads(4); torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
sheet = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True).active
iterator = iter(sheet.values); headers = next(iterator)
assert headers[5] == 'text' and headers[16] == 'Cyberbullying label'
groups, current = [], []
for row in iterator:
    if all(v is None for v in row):
        if current: groups.append(current); current = []
    else: current.append(row)
if current: groups.append(current)
conversations, y, ids = [], [], []
for group in groups:
    labels = {int(r[16]) for r in group if r[16] is not None}
    texts = [r[5].strip() for r in group if isinstance(r[5], str) and r[5].strip()]
    if len(labels) != 1 or not texts or not labels <= {0,1,2}: continue
    conversations.append(texts); y.append(int(next(iter(labels)) in (1,2)))
    # Excel represents IDs as floats: conservative equality can over-group IDs.
    ids.append([str(r[0]) for r in group if r[0] is not None])
y = np.array(y); parent = list(range(len(y)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a,b):
    a,b=find(a),find(b)
    if a!=b: parent[b]=a
seen = {}
for i,(texts,tweet_ids) in enumerate(zip(conversations,ids)):
    for key in [('text', re.sub(r'\s+',' ',text).strip().casefold()) for text in texts] + [('tweet_id', x) for x in tweet_ids]:
        if key in seen: union(i,seen[key])
        else: seen[key]=i
components = np.array([find(i) for i in range(len(y))])
# Fixed split before inference. Development and test are within this already explored corpus.
folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)),y,components))
test = folds[0][1]; dev = folds[1][1]; train = np.array(sorted(set(range(len(y)))-set(test)-set(dev)))
assert not (set(components[train]) & set(components[dev]) or set(components[train]) & set(components[test]) or set(components[dev]) & set(components[test]))
assert all(len(set(y[s])) == 2 for s in (train,dev,test))
tok = AutoTokenizer.from_pretrained(MODEL,revision=REV,local_files_only=True)
texts = [' '.join(f'Message {j+1}: {v}' for j,v in enumerate(c)) for c in conversations]
trunc = np.array([len(tok(t, truncation=False)['input_ids']) > 128 for t in texts])
class Mobile(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=AutoModel.from_pretrained(MODEL,revision=REV,local_files_only=True); self.head=nn.Linear(self.encoder.config.hidden_size,2)
    def forward(self,b):
        h=self.encoder(**b).pooler_output
        return self.head(F.layer_norm(h,(self.encoder.config.hidden_size,)))
m=Mobile()
for p in m.parameters(): p.requires_grad=False
for layer in list(m.encoder.encoder.layer)[-2:]:
    for p in layer.parameters(): p.requires_grad=True
for p in m.head.parameters(): p.requires_grad=True
params={n:p for n,p in m.named_parameters() if p.requires_grad}
opt=torch.optim.AdamW([{'params':[p for n,p in params.items() if n.startswith('encoder.')], 'lr':2e-5},{'params':[p for n,p in params.items() if n.startswith('head.')], 'lr':1e-4}],weight_decay=.01)
weights=torch.tensor([1.0,float((len(train)-sum(y[train]))/sum(y[train]))])
def batch(ix): return tok([texts[i] for i in ix],padding=True,truncation=True,max_length=128,return_tensors='pt')
order=list(train); rng=random.Random(seed); losses=[]
for epoch in range(4):
    rng.shuffle(order); m.train(); m.encoder.eval()
    for layer in list(m.encoder.encoder.layer)[-2:]: layer.train()
    total=0.0
    for off in range(0,len(order),4):
        ix=order[off:off+4]; opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(batch(ix)),torch.tensor(y[ix],dtype=torch.long),weight=weights)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(params.values()),1.0); opt.step(); total+=loss.item()
    losses.append(total/((len(order)+3)//4))
    print('epoch',epoch+1,'loss',round(losses[-1],4),flush=True)
artifact=ROOT/'artifacts/cyby23-bullying-pilot/mobilebert-trainable.safetensors'; artifact.parent.mkdir(parents=True,exist_ok=True)
save_file({n:p.detach().cpu().contiguous() for n,p in params.items()},str(artifact))
def mobile_scores(indices):
    out=[]; m.eval()
    with torch.inference_mode():
        for off in range(0,len(indices),4): out.extend(torch.softmax(m(batch(indices[off:off+4])).float(),dim=1)[:,1].tolist())
    return np.array(out)
# Save scores before releasing trainable model and memory; no changes based on test.
mobile_dev=mobile_scores(dev); mobile_test=mobile_scores(test)
del m,opt,params
asset=ROOT/'app/src/main/assets/model'
toxic_tok=BertTokenizerFast(vocab_file=str(asset/'vocab.txt'),do_lower_case=True)
session=ort.InferenceSession(str(asset/'toxic-bert.onnx'),providers=['CPUExecutionProvider'])
input_names={i.name for i in session.get_inputs()}
def toxicity_scores(indices):
    out=[]
    for i in indices:
        scores=[]
        for message in conversations[i]:
            b=toxic_tok(message,return_tensors='np',truncation=True,max_length=128)
            logit=session.run(None,{k:v.astype(np.int64) for k,v in b.items() if k in input_names})[0][0][0]
            scores.append(float(1/(1+np.exp(-logit))))
        out.append(max(scores))
    return np.array(out)
def stats(labels,scores):
    pred=scores>=.5; tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return {'n':len(labels),'positive':int(sum(labels)),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'precision':precision_score(labels,pred,zero_division=0),'recall':recall_score(labels,pred,zero_division=0),'f1':f1_score(labels,pred,zero_division=0)}
toxic_dev=toxicity_scores(dev); toxic_test=toxicity_scores(test)
dev_results={'mobilebert':stats(y[dev],mobile_dev),'toxic_bert_any_message':stats(y[dev],toxic_dev)}
selected=max(dev_results,key=lambda n:(dev_results[n]['f1'],n))
report={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'seed':seed,'objective':'CYBY23 thread labels: class 0 aggression without bullying vs classes 1 and 2 bullying; no normal safe class',
 'split':'StratifiedGroupKFold, deterministic 5-fold; one fold development, one test, three train; union by normalized exact message and Excel tweet ID; no author fields, fuzzy similarity or reliable temporal split',
 'partitions':{name:{'n':len(idx),'positives':int(sum(y[idx])),'mobilebert_truncated':int(sum(trunc[idx]))} for name,idx in [('train',train),('development',dev),('test',test)]},
 'models':'MobileBERT last 2 layers and new head fine-tuned for 4 epochs on first 128 tokens of each thread; toxic-bert fixed existing 0.5 threshold applied to each message and aggregated with max. Both thresholds fixed at 0.5.',
 'development':dev_results,'selection_policy':'highest development F1; name breaks ties', 'selected_by_development':selected,
 'test':{'mobilebert':stats(y[test],mobile_test),'toxic_bert_any_message':stats(y[test],toxic_test)},'loss_per_epoch':losses,
 'mobilebert_artifact':str(artifact.relative_to(ROOT)),'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
 'limitations':'Small single-source exploratory comparison; corpus already inspected and evaluated before split, so this is not a blinded or independent final test. Author overlap and approximate duplicates not ruled out; unknown pretraining overlap; message-level max vs truncated thread model uses different context. Public Twitter aggression-only sampling, not private chats; no independent expert annotation, real-world safety or Android measurement.'}
DEST.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
