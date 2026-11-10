#!/usr/bin/env python3
"""Exploratory thread-level CYBY23 vs max individual-message toxicity."""
import hashlib, json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import openpyxl
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from transformers import BertTokenizerFast

root = Path(__file__).resolve().parents[1]
source = root/'data/external/cyby23.xlsx'
sheet = openpyxl.load_workbook(source, read_only=True, data_only=True).active
rows = iter(sheet.values)
headers = next(rows)
assert headers[5] == 'text' and headers[16] == 'Cyberbullying label'
groups, current = [], []
for row in rows:
    if all(v is None for v in row):
        if current: groups.append(current); current = []
    else: current.append(row)
if current: groups.append(current)
asset = root/'app/src/main/assets/model'
tok = BertTokenizerFast(vocab_file=str(asset/'vocab.txt'), do_lower_case=True)
session = ort.InferenceSession(str(asset/'toxic-bert.onnx'), providers=['CPUExecutionProvider'])
inputs = {i.name for i in session.get_inputs()}
labels, predicted, lengths, truncated = [], [], [], 0
skipped = 0
for group in groups:
    label_values = {int(row[16]) for row in group if row[16] is not None}
    messages = [str(row[5]) for row in group if isinstance(row[5], str) and row[5].strip()]
    if len(label_values) != 1 or not messages or not label_values <= {0,1,2}:
        skipped += 1; continue
    scores = []
    for text in messages:
        truncated += int(len(tok(text, truncation=False)['input_ids']) > 128)
        batch = tok(text, return_tensors='np', truncation=True, max_length=128)
        logits = session.run(None, {k:v.astype(np.int64) for k,v in batch.items() if k in inputs})[0][0]
        scores.append(float(1/(1+np.exp(-logits[0]))))
    labels.append(int(next(iter(label_values)) in (1,2)))
    predicted.append(int(max(scores)>=0.5))
    lengths.append(len(messages))
tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0,1]).ravel()
result = {'source':'CYBY23, University of Science Malaysia, https://opendata.usm.my/items/f351c23f-009f-4b8f-bba3-563f5fb1bbfd', 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'source_citation':'Alfurayj, Yee and Lutfi, Bystanders Unveiled, TENCON 2023, pp. 1012–1017; Alfurayj and Lutfi, Exploring Bystanders’ Roles, TENCON 2023, pp. 1018–1023', 'protocol':'Each blank-delimited thread: positive if corpus thread label is 1 or 2; flag if any text in thread has toxic-bert toxicity >=0.5. No training or threshold tuning.', 'total_groups':len(groups), 'skipped_groups':skipped, 'n':len(labels), 'positive':sum(labels), 'messages':sum(lengths), 'truncated_messages':truncated, 'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'precision':precision_score(labels,predicted,zero_division=0),'recall':recall_score(labels,predicted,zero_division=0), 'limitations':'Exploratory cross-task aggregate: toxicity per message vs thread bullying label; class 0 is aggression, not normal conversation; public Twitter threads, not private Instagram; repeated responses and thread length affect max score; no author-independence or pretraining-overlap audit; no text or IDs exported; not phone inference.'}
(root/'docs/cyby23-exploratory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
