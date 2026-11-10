#!/usr/bin/env python3
"""Exploratory author-level comparison, NOT a message-level or Instagram benchmark."""
import csv, io, json, random, zipfile
from pathlib import Path
import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from transformers import BertTokenizerFast

root = Path(__file__).resolve().parents[1]
archive = root / 'data/external/dadvar-youtube.zip'
assets = root / 'app/src/main/assets/model'
with zipfile.ZipFile(archive) as z:
    stream = io.TextIOWrapper(z.open('YouTube_Dadvar_2014.csv'), encoding='utf-8-sig', errors='replace', newline='')
    rows = [r for r in csv.DictReader(stream) if r['Class'] in ('0', '1') and r['Comments'].strip()]
rng = random.Random(42)
# Balanced, stratified sample; not representative of prevalence in the original corpus.
selected = []
for label in ('0', '1'):
    pool = [r for r in rows if r['Class'] == label]
    selected.extend(rng.sample(pool, 100))
rng.shuffle(selected)
tok = BertTokenizerFast(vocab_file=str(assets / 'vocab.txt'), do_lower_case=True)
session = ort.InferenceSession(str(assets / 'toxic-bert.onnx'), providers=['CPUExecutionProvider'])
inputs = {x.name for x in session.get_inputs()}
y, scores, truncated = [], [], 0
for row in selected:
    original = row['Comments']
    text = original[:4096]
    truncated += int(len(original) > 4096 or len(tok(text, add_special_tokens=True, truncation=False)['input_ids']) > 128)
    batch = tok(text, return_tensors='np', truncation=True, max_length=128)
    logits = session.run(None, {k: v.astype(np.int64) for k, v in batch.items() if k in inputs})[0][0]
    y.append(int(row['Class']))
    scores.append(float(1 / (1 + np.exp(-logits[0]))))
pred = np.array(scores) >= .5
tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
report = {'dataset': 'Dadvar YouTube 2014, DOI 10.5281/zenodo.10143096', 'sample': '100 per class, random seed 42',
          'unit': 'author history, first 128 tokens only', 'threshold': .5, 'n': len(y), 'truncated_histories': truncated,
          'precision': precision_score(y, pred, zero_division=0), 'recall': recall_score(y, pred, zero_division=0),
          'true_positives': int(tp), 'false_positives': int(fp), 'true_negatives': int(tn), 'false_negatives': int(fn),
          'limitations': 'Exploratory cross-task comparison: author-level bullying labels vs first-window toxicity; balanced sample, source and age differ from Instagram, pretraining overlap unknown. Not a validation of bullying or grooming detection.'}
out = root / 'reports/dadvar-exploratory.json'; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))
