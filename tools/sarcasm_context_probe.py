#!/usr/bin/env python3
"""Compare the same target message alone and after its synthetic context."""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/sarcasm-context-results.json'


def counts(cases, model, variant):
    grouped = {}
    for category in ('friendly_teasing', 'hostile_sarcasm', 'direct_insult'):
        items = [c for c in cases if c['category'] == category]
        positives = sum(c[model][variant]['flagged'] for c in items)
        grouped[category] = {'n': len(items), 'flags': positives, 'not_flagged': len(items)-positives}
    return grouped


def main():
    if DEST.exists():
        raise SystemExit('Report already exists')
    source = ROOT / 'data/sarcasm-context-pairs.json'
    data = json.loads(source.read_text())
    cases = data['cases']
    assert len(cases) == 18 and len({c['id'] for c in cases}) == 18
    assert Counter(c['category'] for c in cases) == {'friendly_teasing': 6, 'hostile_sarcasm': 6, 'direct_insult': 6}
    assert all(c['label'] == (c['category'] != 'friendly_teasing') for c in cases)
    previous = json.loads((ROOT/'docs/cyby24-standardized.json').read_text())['results']
    thresholds = {entry['model']: entry['threshold'] for entry in previous}
    torch.set_num_threads(4)
    bert = BertTokenizerFast(vocab_file=str(ROOT/'app/src/main/assets/model/vocab.txt'), do_lower_case=True)
    roberta_path = ROOT/'.cache/roberta-cyberbullying'
    roberta_tok = AutoTokenizer.from_pretrained(roberta_path, local_files_only=True)
    sys.path.insert(0, os.environ['PHOC_RUNTIME'])
    from phocinae.engine import Engine, QTYPES
    repo = 'Phocinae/Phocinae-Largha-150M-v1'
    revision = 'a57030322df13c8c55ab7a99b8b7b176e2c5aa85'
    path = snapshot_download(repo, revision=revision, local_files_only=True,
                             allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
    engine = Engine(path, device='cpu', warm=False, threads=4)
    question = {'id':'risk','type':'noul','instructions':'Does this conversation contain cyberbullying?'}
    spec = engine._question_spec(question)
    room = 256 - len(engine._build_sequence(spec, [], [0,1], 64, 256)[0])
    weights = ROOT/'artifacts/cyby24-matched/phocinae.safetensors'
    previous_phoc = json.loads((ROOT/'docs/cyby24-matched-phocinae.json').read_text())
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == previous_phoc['artifact']['sha256']
    assert not engine.model.load_state_dict(load_file(str(weights)), strict=False).unexpected_keys
    engine.model.eval()

    def views(case):
        return {'alone': case['target'], 'with_context': case['context'] + '\nA: ' + case['target']}

    for case in cases:
        case['phocinae'] = {}
        for variant, text in views(case).items():
            assert len(bert(text)['input_ids']) <= 256 and len(roberta_tok(text)['input_ids']) <= 256
            assert len(engine.tok.encode(text)) <= room
            ids, markers = engine._build_sequence(spec, engine.tok.encode(text), [0,1], 64, 256)
            assert len(ids) <= 256 and len(markers) == 2
            item = {'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':question}
            b = engine._collate([item])
            with torch.inference_mode():
                logits = engine.model(b['input_ids'], b['attention_mask'], b['marker_pos'], b['marker_mask'], b['qtype'])[0]
                score = float(torch.softmax(logits.float(), dim=1)[0,1])
            case['phocinae'][variant] = {'score':score,'flagged':score >= thresholds['phocinae']}
    del engine

    model = AutoModelForSequenceClassification.from_pretrained(roberta_path, local_files_only=True).eval()
    for case in cases:
        case['roberta'] = {}
        for variant, text in views(case).items():
            with torch.inference_mode():
                score = float(torch.softmax(model(**roberta_tok(text, return_tensors='pt')).logits, dim=1)[0,1])
            case['roberta'][variant] = {'score':score,'flagged':score >= thresholds['roberta']}
    del model

    session = ort.InferenceSession(str(ROOT/'app/src/main/assets/model/toxic-bert.onnx'), providers=['CPUExecutionProvider'])
    input_names = {x.name for x in session.get_inputs()}
    for case in cases:
        case['toxic'] = {}
        for variant, text in views(case).items():
            b = bert(text, return_tensors='np')
            logits = session.run(None, {k:v.astype(np.int64) for k,v in b.items() if k in input_names})[0][0][0]
            score = float(1/(1+np.exp(-logits)))
            case['toxic'][variant] = {'score':score,'flagged':score >= thresholds['toxic'], 'app_threshold_flagged':score >= .5}

    summaries = {model: {variant:counts(cases,model,variant) for variant in ('alone','with_context')}
                 for model in ('toxic','roberta','phocinae')}
    report = {'dataset_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'protocol':'Each target unchanged; compare target alone with three preceding turns plus target. Frozen CYBY24 256-token thresholds; no threshold fitting on these cases.',
              'thresholds':{model: thresholds[model] for model in ('toxic','roberta','phocinae')},
              'limitations':'Author-intended labels, small artificial sample, context can contain independently harmful messages; hostile target alone is ambiguous. Prior CYBY24 thresholds are uncalibrated and task/model definitions differ. Host CPU only.',
              'summary':summaries,'cases':cases}
    DEST.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
