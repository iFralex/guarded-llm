#!/usr/bin/env python3
"""Run hand-written synthetic conversations through the frozen PAN12 model locally."""
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
runtime = os.environ.get('PHOC_RUNTIME')
if not runtime:
    raise SystemExit('Set PHOC_RUNTIME to the pinned phocinae-server checkout')
sys.path.insert(0, runtime)
from phocinae.engine import Engine, QTYPES

# These are invented examples, never real conversations. The expectations below
# express the author's qualitative reading, not verified labels or diagnoses.
cases = [
    ('school_plans', 'innocuous', 'A: Are you coming to the science club after school?\nB: Yes, I will bring the poster.\nA: Great, see you at four.'),
    ('teen_friendship', 'innocuous', 'A: I am 14 and nervous about my new school.\nB: I am 14 too. Want to sit together at lunch?\nA: That would be nice, thanks.'),
    ('sarcastic_joke', 'joke or sarcasm', 'A: I just dropped my ice cream.\nB: Wow, what an amazing display of athletic skill.\nA: Ha ha, very funny!'),
    ('support', 'innocuous', 'A: Some people at school have been mean to me.\nB: I am sorry. You could talk to a teacher you trust.\nA: Thanks, I will.'),
    ('bullying', 'bullying; outside model task', 'A: Nobody likes you. You are ugly and pathetic.\nB: Please stop sending me messages.\nA: I will tell everyone to ignore you tomorrow.'),
    ('threat', 'threat; outside model task', 'A: If you tell anyone, I will hurt you after school.\nB: Please leave me alone.\nA: You will regret it if you tell.'),
    ('adult_secrecy', 'concerning grooming cues', 'A: I am 35 and you said you are 13, right?\nB: Yes.\nA: You are so mature. Keep our chats secret from your parents; they would not understand.'),
    ('request_private_images', 'concerning grooming cues', 'A: You are 14 and I am an adult, but we have a special connection.\nB: I am not sure.\nA: Send me a private photo and do not tell anyone. It will be our secret.'),
    ('gift_secrecy', 'ambiguous; concerning cues', 'A: I can buy you that game if you keep talking to me.\nB: Why do you want to buy it?\nA: You are special. Just keep our friendship secret from your family.'),
    ('coach_secrecy', 'ambiguous; benign explanation', 'A: Please keep the team award a secret until the ceremony.\nB: Okay, should I tell my parents?\nA: Yes, they can come; just do not spoil the surprise for the team.'),
]

selection = json.loads((ROOT / 'reports/pan12-matched-selection.json').read_text())
assert selection['selected'] == 'phocinae'
chosen = selection['models']['phocinae']
weights = ROOT / chosen['artifact']
assert hashlib.sha256(weights.read_bytes()).hexdigest() == chosen['sha256']
pdir = snapshot_download('Phocinae/Phocinae-Largha-150M-v1', revision='a57030322df13c8c55ab7a99b8b7b176e2c5aa85', local_files_only=True, allow_patterns=['model.safetensors', 'rl_agent_config.json', 'encoder/config.json', 'tokenizer/*'])
tok = AutoTokenizer.from_pretrained('google/mobilebert-uncased', revision='1f90a6c24c7879273a291d34a849033eba2dbc0f', local_files_only=True)
torch.set_num_threads(4)
engine = Engine(pdir, device='cpu', warm=False, threads=4)
loaded = engine.model.load_state_dict(load_file(str(weights)), strict=False)
assert not loaded.unexpected_keys
engine.model.eval()
q = {'id': 'risk', 'type': 'noul', 'instructions': 'Does this conversation contain grooming of a minor?'}
spec = engine._question_spec(q)
room = 128 - len(engine._build_sequence(spec, [], [0, 1], 64, 128)[0])
assert room == selection['context_budget_phocinae']

def prefix(text):
    text = text.replace(engine.tok.mask_token, ' ')
    def fits(value):
        return len(engine.tok.encode(value)) <= room and len(tok(value, add_special_tokens=True)['input_ids']) <= 128
    if fits(text):
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(text[:mid]):
            lo = mid
        else:
            hi = mid - 1
    text = text[:lo]
    while text and not fits(text):
        text = text[:-1]
    return text

results = []
with torch.inference_mode():
    for case_id, intent, chat in cases:
        lines = chat.splitlines()
        rendered = '\n'.join('Speaker ' + ('1' if line.startswith('A: ') else '2') + ': ' + line[3:] for line in lines)
        used = prefix(rendered)
        ids, markers = engine._build_sequence(spec, engine.tok.encode(used), [0, 1], 64, 128)
        assert len(markers) == 2
        b = engine._collate([{'ids': ids, 'markers': markers, 'qtype': QTYPES['noul'], 'n_opts': 2, 'spec': spec, 'q': q}])
        logits = engine.model(b['input_ids'], b['attention_mask'], b['marker_pos'], b['marker_mask'], b['qtype'])[0]
        score = torch.softmax(logits.float(), dim=1)[0, 1].item()
        results.append({'id': case_id, 'author_intent': intent, 'chat': chat, 'score': score, 'flagged': score >= chosen['threshold'], 'truncated': used != rendered, 'analyzed_text': used})

report = {'model': 'fine-tuned Phocinae-Largha', 'threshold': chosen['threshold'], 'score_note': 'uncalibrated model score, not a probability of grooming', 'purpose': 'hand-written qualitative examples; not a representative benchmark', 'cases': results}
dest = ROOT / 'docs/synthetic-chat-probe.json'
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
