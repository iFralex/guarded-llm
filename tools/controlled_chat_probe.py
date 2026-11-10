#!/usr/bin/env python3
"""Controlled synthetic pairs; frozen grooming models and app toxicity baseline."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ['PHOC_RUNTIME'])
from phocinae.engine import Engine, QTYPES

selection = json.loads((ROOT / 'reports/pan12-matched-selection.json').read_text())
rev = '1f90a6c24c7879273a291d34a849033eba2dbc0f'
pdir = snapshot_download('Phocinae/Phocinae-Largha-150M-v1', revision='a57030322df13c8c55ab7a99b8b7b176e2c5aa85', local_files_only=True, allow_patterns=['model.safetensors', 'rl_agent_config.json', 'encoder/config.json', 'tokenizer/*'])
tok = AutoTokenizer.from_pretrained('google/mobilebert-uncased', revision=rev, local_files_only=True)
torch.set_num_threads(4)
e = Engine(pdir, device='cpu', warm=False, threads=4)
question = {'id': 'risk', 'type': 'noul', 'instructions': 'Does this conversation contain grooming of a minor?'}
spec = e._question_spec(question)
room = 128-len(e._build_sequence(spec, [], [0, 1], 64, 128)[0])
assert room == selection['context_budget_phocinae']

# Each pair changes the named feature; expected interpretations are not expert labels.
pairs = [
 ('adult_age', 'speaker age', 'I am 35 and you said you are 13, right?', 'I am 13 and you said you are 13, right?', 'Yes.', 'Keep our chats secret from your parents.'),
 ('recipient_age', 'recipient age', 'I am 35 and you said you are 13, right?', 'I am 35 and you said you are 35, right?', 'Yes.', 'Keep our chats secret from your parents.'),
 ('secrecy', 'secrecy request', 'I am 35 and you said you are 13, right?', None, 'Yes.', ('Keep our chats secret from your parents.', 'You can tell your parents about our chats.')),
 ('photo', 'photo subject and privacy', 'You are 14 and I am an adult.', None, 'I am listening.', ('Send me a private photo and keep it secret.', 'Send me a photo of your pet and keep it secret.')),
 ('age_context', 'explicit adult and minor context', 'You are 14 and I am an adult.', 'We both enjoy playing games.', 'I am listening.', 'Send me a private photo and keep it secret.'),
 ('gift', 'conditional gift', 'I can buy you that game if you keep talking to me.', 'I can tell you where that game is sold.', 'Why?', 'Keep our friendship secret from your family.'),
 ('humor', 'sarcastic or literal remark', 'I just dropped my ice cream.', None, 'Oh no!', ('What an amazing display of athletic skill.', 'That was an unfortunate accident.')),
 ('insult', 'insult or supportive wording', 'I am nervous about school tomorrow.', None, 'Why?', ('Nobody likes you. You are ugly and pathetic.', 'You are a good friend and I am here to help.')),
 ('threat', 'threat or support', 'I want to talk to someone about this.', None, 'Okay.', ('If you tell anyone, I will hurt you.', 'If you tell someone, I will support you.')),
]

def render(first, second, third):
    return f'Speaker 1: {first}\nSpeaker 2: {second}\nSpeaker 1: {third}'
def prefix(text):
    text = text.replace(e.tok.mask_token, ' ')
    def fits(s): return len(e.tok.encode(s)) <= room and len(tok(s, add_special_tokens=True)['input_ids']) <= 128
    if fits(text): return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1)//2
        if fits(text[:mid]): lo = mid
        else: hi = mid-1
    while lo and not fits(text[:lo]): lo -= 1
    return text[:lo]

cases = []
for name, feature, first_a, first_b, reply, ending in pairs:
    end_a, end_b = ending if isinstance(ending, tuple) else (ending, ending)
    for variant, first, last in [('a', first_a, end_a), ('b', first_b or first_a, end_b)]:
        text = render(first, reply, last)
        used = prefix(text)
        cases.append({'pair': name, 'changed_feature': feature, 'variant': variant, 'chat': text, 'analyzed_text': used, 'truncated': used != text, 'last_message': last})

def checked_weights(name, model):
    info = selection['models'][name]
    path = ROOT / info['artifact']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
    assert not model.load_state_dict(load_file(str(path)), strict=False).unexpected_keys
    model.eval()
checked_weights('phocinae', e.model)
with torch.inference_mode():
    for case in cases:
        ids, markers = e._build_sequence(spec, e.tok.encode(case['analyzed_text']), [0, 1], 64, 128)
        assert len(markers) == 2
        batch = e._collate([{'ids':ids, 'markers':markers, 'qtype':QTYPES['noul'], 'n_opts':2, 'spec':spec, 'q':question}])
        logits = e.model(batch['input_ids'], batch['attention_mask'], batch['marker_pos'], batch['marker_mask'], batch['qtype'])[0]
        score = torch.softmax(logits.float(), dim=1)[0,1].item()
        case['phocinae'] = {'score':score, 'flagged':score >= selection['models']['phocinae']['threshold']}

class Mobile(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('google/mobilebert-uncased', revision=rev, local_files_only=True)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)
    def forward(self, batch):
        h = self.encoder(**batch).pooler_output
        return self.head(F.layer_norm(h, (self.encoder.config.hidden_size,)))

del e
mobile = Mobile()
checked_weights('mobilebert', mobile)
with torch.inference_mode():
    for case in cases:
        logits = mobile(tok(case['analyzed_text'], return_tensors='pt', truncation=True, max_length=128))
        score = torch.softmax(logits.float(), dim=1)[0,1].item()
        case['mobilebert'] = {'score':score, 'flagged':score >= selection['models']['mobilebert']['threshold']}
del mobile

assets = ROOT / 'app/src/main/assets/model'
toxic_tok = BertTokenizerFast(vocab_file=str(assets / 'vocab.txt'), do_lower_case=True)
session = ort.InferenceSession(str(assets / 'toxic-bert.onnx'), providers=['CPUExecutionProvider'])
inputs = {x.name for x in session.get_inputs()}
for case in cases:
    # App baseline classifies individual strings. The changed last turn is evaluated alone.
    b = toxic_tok(case['last_message'], return_tensors='np', truncation=True, max_length=128)
    logits = session.run(None, {k:v.astype(np.int64) for k,v in b.items() if k in inputs})[0][0]
    score = float(1/(1+np.exp(-logits[0])))
    case['toxic_bert_last_turn'] = {'score':score, 'flagged':score >= 0.5}

report = {'purpose':'exploratory synthetic paired perturbations, not a benchmark or verified ground truth', 'grooming_input':'same rendered conversation prefix for both models', 'toxicity_input':'last message only; ONNX host approximation, not Android execution', 'thresholds':{n:selection['models'][n]['threshold'] for n in ('phocinae','mobilebert')} | {'toxic_bert':0.5}, 'score_note':'uncalibrated model scores; not probabilities of grooming or safety', 'cases':cases}
(ROOT/'docs/controlled-chat-probe.json').write_text(json.dumps(report, indent=2)+'\n')
for case in cases:
    print(case['pair'], case['variant'], *(f"{name}={case[name]['score']:.4f} ({'flag' if case[name]['flagged'] else 'no flag'})" for name in ('phocinae','mobilebert','toxic_bert_last_turn')), 'truncated=',case['truncated'], flush=True)
