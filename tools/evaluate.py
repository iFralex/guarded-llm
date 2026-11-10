#!/usr/bin/env python3
"""Evaluate separate threshold-selection and held-out synthetic datasets."""
import csv
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from transformers import BertTokenizerFast

root = Path(__file__).resolve().parents[1]
assets = root / 'app/src/main/assets/model'
tok = BertTokenizerFast(vocab_file=str(assets / 'vocab.txt'), do_lower_case=True)
session = ort.InferenceSession(str(assets / 'toxic-bert.onnx'), providers=['CPUExecutionProvider'])
def evaluate(path):
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
    y, scores = [], []
    for row in rows:
        batch = tok(row['text'], return_tensors='np', truncation=True, max_length=128)
        logits = session.run(None, {k: v.astype(np.int64) for k, v in batch.items() if k in {i.name for i in session.get_inputs()}})[0][0]
        y.append(int(row['toxic'])); scores.append(float(1 / (1 + np.exp(-logits[0]))))
    return np.array(y), np.array(scores)
# v1 deliberately uses a fixed research threshold. Calibration is reported separately.
report = {'threshold': 0.5, 'warning': 'Synthetic, small research set; not representative of Instagram or grooming.'}
for split in ['calibration', 'test']:
    y, scores = evaluate(root / 'data' / f'{split}.csv')
    pred = scores >= report['threshold']
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    report[split] = {'n': len(y), 'precision': precision_score(y, pred, zero_division=0), 'recall': recall_score(y, pred, zero_division=0), 'false_positives': int(fp), 'false_negatives': int(fn), 'true_positives': int(tp), 'true_negatives': int(tn)}
path = root / 'reports/evaluation.json'; path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report, indent=2))
