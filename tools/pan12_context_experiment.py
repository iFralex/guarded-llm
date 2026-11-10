#!/usr/bin/env python3
"""Exploratory PAN12 author-grouped contextual baseline. Never opens the official test corpus."""
import hashlib
import io
import json
import random
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/external/pan12.zip'
EXPECTED_SHA256 = '1c0efbfc23e38ed52b38ae77600a758f9c1a0c53da825641ad82490b6541b81d'
with ARCHIVE.open('rb') as handle:
    digest = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(handle.read()).hexdigest()
if digest != EXPECTED_SHA256:
    raise SystemExit('Unexpected PAN12 archive checksum')

# Union authors and identical input windows so a participant or exact window cannot cross splits.
class UnionFind:
    def __init__(self): self.parents = {}; self.sizes = {}
    def find(self, x):
        self.parents.setdefault(x, x); self.sizes.setdefault(x, 1)
        while self.parents[x] != x:
            self.parents[x] = self.parents[self.parents[x]]; x = self.parents[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.sizes[a] < self.sizes[b]: a, b = b, a
            self.parents[b] = a; self.sizes[a] += self.sizes[b]

uf = UnionFind()
examples = []
with zipfile.ZipFile(ARCHIVE) as outer:
    matches = [x for x in outer.namelist() if 'training-corpus' in x and x.endswith('.zip')]
    if len(matches) != 1: raise RuntimeError('Expected one training archive')
    with zipfile.ZipFile(io.BytesIO(outer.read(matches[0]))) as train:
        names = train.namelist()
        truth = next(x for x in names if x.endswith('training-corpus-predators-2012-05-01.txt'))
        predators = set(train.read(truth).decode('utf-8-sig').split())
        xml = next(x for x in names if x.endswith('training-corpus-2012-05-01.xml'))
        with train.open(xml) as src:
            for _, node in ET.iterparse(src, events=('end',)):
                if node.tag != 'conversation': continue
                messages = node.findall('message')
                authors = {m.findtext('author', '').strip() for m in messages} - {''}
                # Labels apply to author across the entire segment, not to each line.
                positive = bool(authors & predators)
                selected = messages[:10]
                prefix_authors = {m.findtext('author', '').strip() for m in selected} - {''}
                # Weak positive labels must actually contain a labeled author's turn in the prefix.
                if positive and not (prefix_authors & predators):
                    node.clear(); continue
                parts = []
                for m in selected:
                    t = ''.join(m.find('text').itertext()) if m.find('text') is not None else ''
                    t = re.sub(r'\s+', ' ', t).strip()[:512]
                    if t: parts.append(t)
                text = '\n'.join(parts)
                if not text or not authors:
                    node.clear(); continue
                key = 'a:' + next(iter(sorted(authors)))
                uf.find(key)
                for author in sorted(authors): uf.union(key, 'a:' + author)
                text_key = 't:' + hashlib.sha256(text.casefold().encode()).hexdigest()
                uf.union(key, text_key)
                examples.append((key, text, int(positive), len(selected)))
                node.clear()
texts = [x[1] for x in examples]
labels = [x[2] for x in examples]
groups = [uf.find(x[0]) for x in examples]
counts = Counter(groups)
print(f'Usable segments: {len(examples)}; positives: {sum(labels)}; author/text components: {len(counts)}; largest component: {max(counts.values())}', flush=True)
# Choose a deterministic, approximately 20% group holdout without using text or predictions.
# Preserve both classes in train and development; use 100 candidate group splits, fixed seed.
import numpy as np
y = np.asarray(labels)
group_array = np.asarray(groups)
splitter = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
best = None
for tr, dev in splitter.split(np.zeros(len(y)), y, group_array):
    if len(set(y[tr])) < 2 or len(set(y[dev])) < 2: continue
    score = abs(len(dev)/len(y) - .2) + abs(float(y[dev].mean()) - float(y.mean()))
    if best is None or score < best[0]: best = (score, tr, dev)
if best is None: raise SystemExit('No valid group-disjoint split; no experiment run')
_, train_idx, dev_idx = best
assert not set(group_array[train_idx]) & set(group_array[dev_idx])
print(f'Train: {len(train_idx)} (positive {int(y[train_idx].sum())}); dev: {len(dev_idx)} (positive {int(y[dev_idx].sum())})', flush=True)
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=100000, sublinear_tf=True, dtype=np.float32)
Xtrain = vectorizer.fit_transform([texts[i] for i in train_idx])
Xdev = vectorizer.transform([texts[i] for i in dev_idx])
model = LogisticRegression(class_weight='balanced', max_iter=300, random_state=42, solver='liblinear')
model.fit(Xtrain, y[train_idx])
prob = model.predict_proba(Xdev)[:, 1]
pred = prob >= .5
tn, fp, fn, tp = confusion_matrix(y[dev_idx], pred, labels=[0,1]).ravel()
report = {
 'dataset': 'PAN12 training corpus only', 'seed': 42,
 'unit': 'conversation segment prefix, up to first 10 messages',
 'label': 'weak segment label from presence of labeled author, requiring a labeled author turn in prefix; not a message-level label',
 'grouping': 'connected components of all authors in full segments and identical normalized input windows',
 'group_count': len(counts), 'largest_group_size': max(counts.values()),
 'split_selection': '100 fixed-seed group splits; choose closest to 20% and full-training prevalence without predictions',
 'train_segments': len(train_idx), 'train_positive': int(y[train_idx].sum()),
 'development_segments': len(dev_idx), 'development_positive': int(y[dev_idx].sum()),
 'model': 'word unigram/bigram TF-IDF and balanced logistic regression',
 'threshold': .5, 'precision': precision_score(y[dev_idx], pred, zero_division=0),
 'recall': recall_score(y[dev_idx], pred, zero_division=0), 'f1': f1_score(y[dev_idx], pred, zero_division=0),
 'average_precision': average_precision_score(y[dev_idx], prob),
 'true_positives': int(tp), 'false_positives': int(fp), 'true_negatives': int(tn), 'false_negatives': int(fn),
 'limitations': 'Research-only offline baseline; author-derived weak segment labels, arbitrary 10-turn prefix, source/decoy artifacts, possible near-duplicate leakage, low-prevalence and platform shift; not real-time or grooming safety validation. No evaluation on official test. No text or author identifiers in report.'
}
out = ROOT / 'reports/pan12-context-development.json'; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))
