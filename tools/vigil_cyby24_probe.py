#!/usr/bin/env python3
"""Exploratory frozen-model comparison on a balanced sample of prior CYBY24 test split."""
import argparse, hashlib, io, json, random, re, zipfile
from collections import Counter
from pathlib import Path
from safecircle_probe import vigilant, file_sha, HASHES
import numpy as np
import openpyxl
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--model',required=True); parser.add_argument('--cli',required=True); args=parser.parse_args()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
assert file_sha(args.model)==HASHES['vigil']
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
sample=rng.sample([int(i) for i in test if y[i]==1],30)+rng.sample([int(i) for i in test if y[i]==0],30)
rng.shuffle(sample)
out={'model':'safecircleai/vigil f16 GGUF','revision':'b239313708157db33314ecc81269e9babc48452a','model_sha256':file_sha(args.model),'sample':'identical seed-43 balanced 60-thread sample previously tested with Gemma, 30 positive and 30 negative','method':'publisher thresholds unchanged, compare any risk flag and bullying category flag separately to thread label','limitations':'Previously explored public Twitter dataset, balanced sample; weak mapping from thread bullying label to any risk; possible author/near-duplicate/pretraining overlap; no raw text or identifiers retained; not Android inference.'}
results={'any':Counter(),'bullying':Counter()}; category_counts=Counter(); shortened=0; errors=0
for step,i in enumerate(sample):
    text='\n'.join(f'Message {j+1}: {v}' for j,v in enumerate(convs[i]));shortened+=int(len(text)>3000);text=text[:3000]
    try:
        output=vigilant(args.cli,args.model,text); flags=output['flagged']
        for k in flags:category_counts[k]+=1
        for measure,pred in [('any',bool(flags)),('bullying','bullying' in flags)]:
            results[measure][{(1,True):'tp',(1,False):'fn',(0,True):'fp',(0,False):'tn'}[(int(y[i]),pred)]]+=1
    except Exception:
        errors+=1
        for measure in results:results[measure]['error_positive' if y[i] else 'error_negative']+=1
    if (step+1)%10==0:print('evaluated',step+1,'/',len(sample),flush=True)
out['results']={k:dict(v) for k,v in results.items()};out['flagged_category_counts']=dict(category_counts);out['truncated_by_character_cap']=shortened;out['errors']=errors
(ROOT/'docs/vigil-cyby24-probe.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
