#!/usr/bin/env python3
"""Exploratory frozen-model comparison on a balanced sample of prior CYBY24 test split."""
import argparse, hashlib, io, json, random, re, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import openpyxl, torch
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--model-dir',required=True); args=parser.parse_args()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
assert sha(Path(args.model_dir)/'model.safetensors')=='9cf3d24e3d64cbad3778219fb24a3e0390a523b50b878f41a99ca0b3656e57df'
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
torch.set_num_threads(4)
tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
model=AutoModelForSequenceClassification.from_pretrained(args.model_dir,local_files_only=True).eval()
assert model.config.num_labels==2

def scores(texts):
    out=[]
    with torch.inference_mode():
        for off in range(0,len(texts),8):
            batch=tok(texts[off:off+8],padding=True,truncation=True,max_length=128,return_tensors='pt')
            out.extend(torch.softmax(model(**batch).logits.float(),dim=1)[:,1].tolist())
    return out

def stats(pred):
    c=Counter()
    for i,p in zip(sample,pred): c[{(1,True):'tp',(1,False):'fn',(0,True):'fp',(0,False):'tn'}[(int(y[i]),p>=.5)]]+=1
    return {k:c[k] for k in ('tp','fp','tn','fn')}

threads=[' '.join(f'Message {j+1}: {v}' for j,v in enumerate(convs[i])) for i in sample]
thread_scores=scores(threads)
messages=[];spans=[]
for i in sample:
    start=len(messages);messages.extend(convs[i]);spans.append((start,len(messages)))
message_scores=scores(messages);max_scores=[max(message_scores[a:b]) for a,b in spans]
out={'model':'nayan90k/roberta-finetuned-cyberbullying-detection','revision':'a5c86fdb6aa46b7d41261907e6c6ecbb2cf66907','sha256':sha(Path(args.model_dir)/'model.safetensors'),'threshold':.5,'sample':'same seed-43 balanced 60-thread CYBY24 subset used for Gemma, Vigil, and MobileBERT; 30 positives and 30 negatives', 'thread_first_128':stats(thread_scores), 'any_message':stats(max_scores),'thread_truncated':sum(len(tok(t,truncation=False)['input_ids'])>128 for t in threads),'limitations':'Exploratory previously inspected public Twitter corpus, not independent or private Instagram chats; source and pretraining overlap unknown. Model trained primarily on individual messages; thread prediction and maximum-per-message are different cross-task proxies. No calibration, phone inference or network behavior measured.'}
(ROOT/'docs/roberta-cyby24-probe.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
# Existing invented examples. Only obvious bullying cases and benign examples are scored; grooming and threats are out of task.
old=json.loads((ROOT/'docs/synthetic-chat-probe.json').read_text())['cases']; held=json.loads((ROOT/'data/gemma4-prompt-holdout.json').read_text())['cases']
examples=[]
for c in old:
    label=1 if c['id']=='bullying' else (0 if c['id'] in ('school_plans','teen_friendship','sarcastic_joke','support','coach_secrecy') else None)
    examples.append({'id':'prior_'+c['id'],'text':c['chat'],'expected':label})
for c in held:
    label=1 if c['id'].startswith('b') else (0 if c['id'].startswith('n') else None)
    examples.append({'id':'holdout_'+c['id'],'text':c['chat'],'expected':label})
scores_all=scores([c['text'] for c in examples]); counts=Counter()
for c,p in zip(examples,scores_all):
    c['score']=p;c['flagged']=p>=.5
    if c['expected'] is None:counts['out_of_task']+=1
    else:counts[{(1,True):'tp',(1,False):'fn',(0,True):'fp',(0,False):'tn'}[(c['expected'],c['flagged'])]]+=1
syn={'model':out['model'],'revision':out['revision'],'threshold':.5,'method':'whole synthetic chat, first 128 RoBERTa tokens; author labels; grooming and threats excluded from scoring','cases':examples,'counts':dict(counts),'limitations':'Previously seen hand-written examples, not independent or representative; model trained on individual posts.'}
(ROOT/'docs/roberta-synthetic-probe.json').write_text(json.dumps(syn,indent=2)+'\n')
print('synthetic',dict(counts),flush=True)
