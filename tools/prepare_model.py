#!/usr/bin/env python3
"""Export the pinned BERT checkpoint and check ONNX against PyTorch outputs."""
from pathlib import Path
import json
import base64
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'app/src/main/assets/model'
MODEL = 'unitary/toxic-bert'
# Pin an immutable revision with --revision when supplied; default resolves once and records it.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--revision', default='main')
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=args.revision)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=args.revision).eval()
revision = model.config._commit_hash
if not revision:
    raise RuntimeError('Could not determine immutable model revision')
vocab = tokenizer.save_pretrained(OUT)
if not (OUT / 'vocab.txt').exists():
    raise RuntimeError('Expected BERT vocabulary')
class Wrapper(torch.nn.Module):
    def __init__(self, wrapped):
        super().__init__(); self.wrapped = wrapped
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.wrapped(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
wrapper = Wrapper(model).eval()
samples = ['You are a kind person.', 'I hate you!', 'Héllo, world! Unkind words.', 'You are so funny 😂']
dummy = tokenizer(samples[0], return_tensors='pt')
path = OUT / 'toxic-bert.onnx'
torch.onnx.export(wrapper, (dummy['input_ids'], dummy['attention_mask'], dummy['token_type_ids']), str(path),
                  input_names=['input_ids', 'attention_mask', 'token_type_ids'], output_names=['logits'],
                  dynamic_axes={name: {0: 'batch', 1: 'sequence'} for name in ['input_ids', 'attention_mask', 'token_type_ids']} | {'logits': {0: 'batch'}},
                  opset_version=17, do_constant_folding=True)
vectors = ROOT / 'app/src/test/resources/tokenizer-vectors.tsv'
vectors.parent.mkdir(parents=True, exist_ok=True)
vectors.write_text(''.join(base64.b64encode(sample.encode()).decode() + '\t' + ','.join(map(str, tokenizer(sample, truncation=True, max_length=128)['input_ids'])) + '\n' for sample in samples))
session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
max_error = 0.0
for sample in samples:
    batch = tokenizer(sample, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad(): expected = wrapper(**batch).numpy()
    actual = session.run(None, {key: value.numpy() for key, value in batch.items() if key in {i.name for i in session.get_inputs()}})[0]
    max_error = max(max_error, float(np.max(np.abs(expected - actual))))
if max_error > 1e-4: raise RuntimeError(f'Export mismatch: {max_error}')
(OUT / 'model-info.json').write_text(json.dumps({'model': MODEL, 'revision': revision, 'license': 'Apache-2.0', 'max_export_error': max_error}, indent=2) + '\n')
for name in ['tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.json']:
    (OUT / name).unlink(missing_ok=True)
print(f'Export verified; revision={revision}, max absolute logit error={max_error:.8f}, bytes={path.stat().st_size}')
