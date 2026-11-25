#!/usr/bin/env python3
"""Preliminary local test of the authors' first eSPD TFLite checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-file', default='model.tflite')
    parser.add_argument('--model-name', default='Vogt et al. BERT VTPAN, non-quantized TFLite')
    parser.add_argument('--archive-verified', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/espd-synthetic.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    model_path = args.model_dir / args.model_file
    digest = hashlib.sha256()
    with model_path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    labels = (args.model_dir / 'labels.txt').read_text().splitlines()
    tokenizer = BertTokenizer(vocab_file=str(args.model_dir / 'vocab'), do_lower_case=True)
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inputs = {}
    for item in interpreter.get_input_details():
        for name in ('input_word_ids', 'input_mask', 'input_type_ids'):
            if name in item['name']:
                inputs[name] = item['index']
    if len(inputs) != 3:
        raise ValueError('Unexpected TFLite input names')
    output = interpreter.get_output_details()[0]['index']
    cases = json.loads((ROOT / 'data/gemma4-prompt-holdout.json').read_text())['cases']
    results = []
    for item in cases:
        batch = tokenizer(item['chat'], max_length=512, truncation=True, padding='max_length', return_tensors='np')
        for name, key in (('input_word_ids', 'input_ids'), ('input_mask', 'attention_mask'), ('input_type_ids', 'token_type_ids')):
            interpreter.set_tensor(inputs[name], batch[key].astype(np.int32))
        interpreter.invoke()
        scores = interpreter.get_tensor(output)[0].tolist()
        results.append({'id': item['id'], 'researcher_label': item['label'], 'scores': scores, 'argmax_label': labels[int(np.argmax(scores))]})
        print(f'Completed {len(results)}/{len(cases)}', flush=True)
    report = {'model': args.model_name,
              'sha256': digest.hexdigest(), 'labels': labels, 'cases': results,
              'limitations': ('Model source archive verification not established. ' if not args.archive_verified else '') + 'Preprocessing is assumed BERT uncased and has not been checked against the original eSPD pipeline. Previously examined researcher-authored synthetic examples; author labels differ from grooming, bullying and threat labels.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
