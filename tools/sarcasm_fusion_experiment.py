#!/usr/bin/env python3
"""Exploratory sarcasm-feature ablation on the previously used CYBY24 split."""
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark_metrics import calibration_threshold, summarize
from cyby24_common_input import build
from cyby24_split import ROOT, load

IRONY_PATH = Path('/Volumes/ORICO/guarded-llm-models/twitter-irony')
IRONY_SHA = '9c82ae7d8e6639f22c7a91d1fab78b880a9862d71d2c9f1fa68d79da1d644bad'
IRONY_REV = '3bf8f118bdf6b00c99658151ef10c9a0b9afd6bf'
DEST = ROOT/'docs/sarcasm-fusion-cyby24.json'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def preprocess(text):
    return ' '.join('@user' if part.startswith('@') and len(part)>1 else 'http' if part.startswith('http') else part
                    for part in text.split(' '))


def score_report(labels, predictions):
    scores = np.asarray(predictions, dtype=float)
    threshold = calibration_threshold(list(zip(labels['calibration'], scores[labels['indices_calibration']])), minimum_recall=.8)
    report = {'threshold': threshold}
    for split in ('train', 'calibration', 'evaluation'):
        indices = labels['indices_'+split]
        report[split] = summarize([(int(y), bool(s >= threshold))
                                   for y,s in zip(labels[split], scores[indices])])
    return report


def main():
    if DEST.exists():
        raise SystemExit('Report already exists')
    if digest(IRONY_PATH/'pytorch_model.bin') != IRONY_SHA:
        raise SystemExit('Irony checkpoint hash mismatch')
    torch.set_num_threads(4)
    conversations, targets, train, calibration, evaluation, _ = load()
    indices = [int(i) for part in (train,calibration,evaluation) for i in part]
    irony_tok = AutoTokenizer.from_pretrained(IRONY_PATH,local_files_only=True)
    def irony_with_preprocessing(text, **kwargs):
        return irony_tok(preprocess(text), **kwargs)
    prefix, cuts, room = build(conversations,indices,extra_tokenizers=(irony_with_preprocessing,))
    texts = [prefix[i] for i in indices]
    synthetic = json.loads((ROOT/'data/sarcasm-context-pairs.json').read_text())['cases']
    synthetic_texts = [case['context'] + '\nA: ' + case['target'] for case in synthetic]
    n_train, n_cal = len(train), len(calibration)
    labels = {'indices_train':slice(0,n_train), 'indices_calibration':slice(n_train,n_train+n_cal),
              'indices_evaluation':slice(n_train+n_cal,len(indices))}
    for split in ('train','calibration','evaluation'):
        labels[split] = np.asarray([int(targets[i]) for i in indices[labels['indices_'+split]]])
    roberta = ROOT/'.cache/roberta-cyberbullying'
    tok = AutoTokenizer.from_pretrained(roberta,local_files_only=True)
    for text in synthetic_texts:
        if len(tok(text)['input_ids']) > 256 or len(irony_tok(preprocess(text))['input_ids']) > 256:
            raise RuntimeError('Synthetic context exceeds a model input window')
    texts.extend(synthetic_texts)
    model = AutoModelForSequenceClassification.from_pretrained(roberta,local_files_only=True).eval()
    embeddings, bullying_scores = [], []
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0,len(texts),8):
            batch = tok(texts[offset:offset+8],padding=True,truncation=True,max_length=256,return_tensors='pt')
            result = model(**batch,output_hidden_states=True)
            embeddings.append(result.hidden_states[-1][:,0,:].cpu().numpy())
            bullying_scores.extend(torch.softmax(result.logits.float(),dim=1)[:,1].cpu().numpy().tolist())
            if offset%200==0:print('bullying model',offset,'/',len(texts),flush=True)
    embeddings = np.concatenate(embeddings).astype(np.float32)
    bullying_scores = np.asarray(bullying_scores)
    del model
    gc.collect()
    irony = AutoModelForSequenceClassification.from_pretrained(IRONY_PATH,local_files_only=True).eval()
    assert irony.config.id2label[1].lower() == 'irony'
    irony_scores = []
    with torch.inference_mode():
        for offset in range(0,len(texts),8):
            batch = irony_tok([preprocess(t) for t in texts[offset:offset+8]],
                              padding=True,truncation=True,max_length=256,return_tensors='pt')
            result = irony(**batch)
            irony_scores.extend(torch.softmax(result.logits.float(),dim=1)[:,1].cpu().numpy().tolist())
            if offset%200==0:print('irony model',offset,'/',len(texts),flush=True)
    irony_scores = np.asarray(irony_scores)
    del irony
    gc.collect()
    if not np.isfinite(embeddings).all() or not np.isfinite(irony_scores).all():
        raise RuntimeError('Non-finite model outputs')
    # Embeddings and scores remain in memory and are never written to disk.
    plain = np.column_stack((embeddings,bullying_scores))
    with_irony = np.column_stack((plain,irony_scores))
    results = {'classifier': score_report(labels,bullying_scores)}
    model_scores = {'classifier':bullying_scores}
    for name,features in (('mlp_without_irony',plain),('mlp_with_irony',with_irony)):
        head = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,16), activation='relu',
                              solver='adam', alpha=.01, batch_size=64, learning_rate_init=.001,
                              max_iter=100, early_stopping=False, random_state=42))
        head.fit(features[labels['indices_train']],labels['train'])
        scores = head.predict_proba(features)[:,1]
        model_scores[name] = scores
        results[name] = score_report(labels,scores)
        results[name]['iterations'] = int(head[-1].n_iter_)
        results[name]['final_loss'] = float(head[-1].loss_)
        results[name]['loss_curve_last_ten'] = [float(x) for x in head[-1].loss_curve_[-10:]]
    synthetic_cases = []
    for offset,case in enumerate(synthetic):
        i = len(indices)+offset
        synthetic_cases.append({'id':case['id'],'category':case['category'],'label':case['label'],
            'irony_score':float(irony_scores[i]),
            'decisions':{name:{'score':float(scores[i]),'flagged':bool(scores[i]>=results[name]['threshold'])}
                         for name,scores in model_scores.items()}})
    report = {
        'objective':'Add an independently trained irony score to bullying embeddings and compare against a same-architecture ablation.',
        'dataset':'CYBY24 grouped split already explored; labels 3/4 positive, 1/2 negative.',
        'partitions':{split:{'n':len(labels[split]),'positive':int(sum(labels[split]))} for split in ('train','calibration','evaluation')},
        'truncated':{split:int(sum(cuts[i] for i in indices[labels['indices_'+split]])) for split in ('train','calibration','evaluation')},
        'input':'same raw thread prefix <=256 tokens for all tokenizers; irony uses publisher username/URL preprocessing',
        'irony_model':'cardiffnlp/twitter-roberta-base-irony',
        'irony_revision':IRONY_REV,'irony_sha256':IRONY_SHA,
        'irony_license':'Model card does not state a license; further rights review is needed before redistribution.',
        'bullying_model':'nayan90k/roberta-finetuned-cyberbullying-detection',
        'bullying_revision':'a5c86fdb6aa46b7d41261907e6c6ecbb2cf66907',
        'bullying_sha256':'9cf3d24e3d64cbad3778219fb24a3e0390a523b50b878f41a99ca0b3656e57df',
        'threshold_policy':'Maximize calibration precision with recall >=0.8, separately per model; thresholds are uncalibrated.',
        'heads':'Same fixed architecture 32/16 ReLU, StandardScaler fit only on train, Adam, seed 42, max 100 iterations; no tuning on evaluation.',
        'results':results, 'synthetic_sarcasm_cases':synthetic_cases,
        'synthetic_limitations':'18 previously explored invented cases and author-intended categories; contexts contain explicit harmful conduct; no independent sarcasm labels.',
        'host_elapsed_seconds':round(time.monotonic()-started,2),
        'limitations':'Previously explored CYBY24, Twitter threads not Instagram; exact duplicate and tweet ID grouping only; author, near-duplicate and pretrained model overlap unknown. No sarcasm labels in CYBY24; irony may measure other cues. Single seed and no Android inference or battery measurements. Not the paper’s original CNN-LSTM method.'
    }
    DEST.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
