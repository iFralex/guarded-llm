"""Deterministic, in-memory PAN12 training-only pilot partitions."""

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

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'data/external/pan12.zip'
DIGEST = '1c0efbfc23e38ed52b38ae77600a758f9c1a0c53da825641ad82490b6541b81d'
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == DIGEST
MODEL = 'google/mobilebert-uncased'
REVISION = '1f90a6c24c7879273a291d34a849033eba2dbc0f'
RNG = random.Random(42)
class UF:
    def __init__(self): self.parent={}; self.size={}
    def find(self, x):
        self.parent.setdefault(x,x); self.size.setdefault(x,1)
        while self.parent[x] != x:
            self.parent[x]=self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a != b:
            if self.size[a] < self.size[b]: a,b=b,a
            self.parent[b]=a; self.size[a]+=self.size[b]

def render(msgs, n):
    roles={}; parts=[]
    for author,text in msgs[:n]:
        if not text: continue
        if author not in roles: roles[author]=len(roles)+1
        parts.append(f'Speaker {roles[author]}: {text}')
    return '\n'.join(parts)

uf=UF(); rows=[]
with zipfile.ZipFile(ARCHIVE) as outer:
    archive=next(x for x in outer.namelist() if 'training-corpus' in x and x.endswith('.zip'))
    with zipfile.ZipFile(io.BytesIO(outer.read(archive))) as train:
        names=train.namelist()
        positives=set(train.read(next(x for x in names if x.endswith('training-corpus-predators-2012-05-01.txt'))).decode('utf-8-sig').split())
        name=next(x for x in names if x.endswith('training-corpus-2012-05-01.xml'))
        with train.open(name) as source:
            for _,node in ET.iterparse(source,events=('end',)):
                if node.tag != 'conversation': continue
                raw=node.findall('message')
                authors={m.findtext('author','').strip() for m in raw}-{''}
                positive=bool(authors & positives)
                # Evaluate the same cohort for all window lengths, with a labeled-author turn by message five.
                early={m.findtext('author','').strip() for m in raw[:5]}-{''}
                if not authors or (positive and not (early & positives)):
                    node.clear(); continue
                messages=[]
                for m in raw[:20]:
                    e=m.find('text'); text=''.join(e.itertext()) if e is not None else ''
                    text=re.sub(r'\s+',' ',text).strip()[:512]
                    messages.append((m.findtext('author','').strip(),text))
                text=render(messages,10)
                if not text: node.clear(); continue
                key='a:'+sorted(authors)[0]
                for a in sorted(authors): uf.union(key,'a:'+a)
                # Each evaluated window's exact duplicates must stay in the same component.
                for window in (5,10,20):
                    digest=hashlib.sha256(render(messages,window).casefold().encode()).hexdigest()
                    uf.union(key,'t:'+digest)
                rows.append((key,messages,int(positive)))
                node.clear()
y=np.array([r[2] for r in rows]); groups=np.array([uf.find(r[0]) for r in rows]); ix=np.arange(len(rows))
# Split whole connected components before selecting resource-bounded samples.
tr,rem=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=42).split(ix,y,groups))
cal_rel,dev_rel=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=43).split(rem,y[rem],groups[rem]))
cal,dev=rem[cal_rel],rem[dev_rel]
assert not (set(groups[tr]) & set(groups[cal]) or set(groups[tr]) & set(groups[dev]) or set(groups[cal]) & set(groups[dev]))
def sample(indices, pos, neg):
    p=[int(i) for i in indices if y[i]==1]; n=[int(i) for i in indices if y[i]==0]
    return RNG.sample(p,min(pos,len(p)))+RNG.sample(n,min(neg,len(n)))
tr=sample(tr,int(os.getenv('PAN12_TRAIN_POS','120')),int(os.getenv('PAN12_TRAIN_NEG','240')))
cal=sample(cal,int(os.getenv('PAN12_CAL_POS','35')),int(os.getenv('PAN12_CAL_NEG','175')))
dev=sample(dev,int(os.getenv('PAN12_DEV_POS','35')),int(os.getenv('PAN12_DEV_NEG','175')))
# Exclude near-duplicates across selected partitions using 3-word shingles, Jaccard >= .8.
def shingles(i):
    words=re.findall(r'\w+',render(rows[i][1],10).lower())
    return set(zip(words,words[1:],words[2:]))
index=defaultdict(set); representations={}
def add_ref(i):
    sh=representations.setdefault(i,shingles(i))
    if len(sh)>=8:
        for gram in sh: index[gram].add(i)
def clean(indices):
    kept=[]; dropped=0
    for i in indices:
        sh=representations.setdefault(i,shingles(i)); candidates=set()
        if len(sh)>=8:
            for gram in sh: candidates.update(index.get(gram,()))
            if any(len(sh & representations[j])/len(sh | representations[j]) >= .8 for j in candidates): dropped+=1; continue
        kept.append(i)
    return kept,dropped
for i in tr: add_ref(i)
cal,drop_cal=clean(cal)
for i in cal: add_ref(i)
dev,drop_dev=clean(dev)
assert len(set(y[cal]))==2 and len(set(y[dev]))==2 and len(set(y[tr]))==2
print('Pilot train/calibration/development:',len(tr),len(cal),len(dev),'near duplicates dropped:',drop_cal,drop_dev,flush=True)

