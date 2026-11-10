#!/usr/bin/env python3
"""Exploratory frozen-model comparison on a balanced sample of prior CYBY24 test split."""
import argparse, hashlib, io, json, random, re, zipfile
from collections import Counter
from pathlib import Path
import litert_lm as lm
import numpy as np
import openpyxl
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--model',required=True); parser.add_argument('--backend',choices=['cpu','gpu'],default='cpu'); args=parser.parse_args()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
assert sha(args.model)=='181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c'
archive=ROOT/'data/external/cyby24.zip'; assert sha(archive)=='fad01343cc62e58558101ff15d27bb484efe38ac1cfe31930aab55dd490a3be2'
with zipfile.ZipFile(archive) as z: sheet=openpyxl.load_workbook(io.BytesIO(z.read('CYBY24.xlsx')),read_only=True,data_only=True).active
it=iter(sheet.values); headers=next(it); assert headers[5]=='text' and headers[7]=='cyberbullying_label'
groups=[]; current=[]
for row in it:
    if all(v is None or (isinstance(v,str) and not v.strip()) for v in row):
        if current: groups.append(current); current=[]
    else: current.append(row)
if current: groups.append(current)
old_rows=list(openpyxl.load_workbook(ROOT/'data/external/cyby23.xlsx',read_only=True,data_only=True).active.values)[1:]
norm=lambda s:re.sub(r'\s+',' ',s).strip().casefold()
old_text={norm(r[5]) for r in old_rows if isinstance(r[5],str) and r[5].strip()}; old_ids={str(r[0]) for r in old_rows if r[0] is not None}
convs=[]; labels=[]; ids=[]
for group in groups:
    lab={int(r[7]) for r in group if r[7] is not None and str(r[7]).strip()}
    texts=[r[5].strip() for r in group if isinstance(r[5],str) and r[5].strip()]
    if len(lab)!=1 or not texts or not lab<={1,2,3,4}: continue
    if any(norm(t) in old_text for t in texts) or any(str(r[0]) in old_ids for r in group if r[0] is not None): continue
    convs.append(texts); labels.append(int(next(iter(lab)) in (3,4))); ids.append([str(r[0]) for r in group if r[0] is not None])
y=np.array(labels); parent=list(range(len(y)))
def find(i):
    while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
    return i
def union(a,b):
    a,b=find(a),find(b)
    if a!=b: parent[b]=a
seen={}
for i,(texts,tids) in enumerate(zip(convs,ids)):
    for key in [('text',norm(t)) for t in texts]+[('tweet_id',v) for v in tids]:
        if key in seen: union(i,seen[key])
        else: seen[key]=i
components=np.array([find(i) for i in range(len(y))])
test=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=67).split(np.zeros(len(y)),y,components))[0][1]
assert len(test)==551 and sum(y[test])==112
rng=random.Random(43)
old=set(rng.sample([int(i) for i in test if y[i]==1],30)+rng.sample([int(i) for i in test if y[i]==0],30))
rng=random.Random(71)
sample=rng.sample([int(i) for i in test if y[i]==1 and i not in old],30)+rng.sample([int(i) for i in test if y[i]==0 and i not in old],30)
rng.shuffle(sample)
assert len(sample)==60 and not (set(sample)&old)
experiment=json.loads((ROOT/'docs/gemma4-prompt-experiment.json').read_text()); selected=experiment['selected']
report={'purpose':'60 additional CYBY24 threads, disjoint from earlier Gemma sample; exploratory, not independently sourced or representative', 'sampling':'seed 71; 30 positive, 30 negative; excludes earlier seed-43 sample', 'model_sha256':sha(args.model),'backend':args.backend,'selected_on':'nine previously observed synthetic examples', 'label_limitations':'Original thread labels are bullying vs normal or aggression; prompts classify broader concern including threats and grooming, so labels and task do not exactly match. Previously inspected dataset, unknown pretraining overlap, incomplete author/near-duplicate audit; no source text, per-case labels or IDs saved.', 'results':{}}
backend=lm.Backend.CPU(thread_count=4) if args.backend=='cpu' else lm.Backend.GPU()
with lm.Engine(args.model,backend=backend,max_num_tokens=2048,cache_dir=None) as engine:
    for name in dict.fromkeys(['baseline',selected]):
        result=Counter(); truncated=0
        for step,i in enumerate(sample):
            text='\n'.join(f'Message {j+1}: {v}' for j,v in enumerate(convs[i]))
            truncated+=int(len(text)>3000); text=text[:3000]
            try:
                with engine.create_conversation(system_message=experiment['prompts'][name],sampler_config=lm.SamplerConfig(top_k=1,temperature=0.0,seed=42),thinking_config=lm.ThinkingConfig(enable_thinking=False,thinking_token_budget=0),max_output_tokens=16) as c:
                    answer=str(c.send_message('CHAT START\n'+text+'\nCHAT END',max_output_tokens=16)).strip()
                match=re.fullmatch(r'\s*(CONCERN|NO_CONCERN|UNCERTAIN)[.!\s]*',answer,re.I)
                pred=match.group(1).upper() if match else 'INVALID'
            except Exception: pred='ERROR'
            label=int(y[i])
            if pred=='CONCERN': result['tp' if label else 'fp']+=1
            elif pred=='NO_CONCERN': result['fn' if label else 'tn']+=1
            else:
                result['abstain_positive' if label else 'abstain_negative']+=1
                result['format_errors' if pred in ('INVALID','ERROR') else 'uncertain']+=1
            if (step+1)%15==0: print(name,step+1,'/',len(sample),flush=True)
        report['results'][name]={'counts':dict(result),'character_cap_truncated_threads':truncated}
        print('completed',name,dict(result),flush=True)
(ROOT/'docs/gemma4-prompt-cyby24.json').write_text(json.dumps(report,indent=2)+'\n')
