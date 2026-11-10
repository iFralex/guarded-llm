#!/usr/bin/env python3
"""Reload selected weights and reproduce development metrics before opening the test archive."""
import os
os.environ.update({'PAN12_TRAIN_POS':'400','PAN12_TRAIN_NEG':'800','PAN12_CAL_POS':'100','PAN12_CAL_NEG':'500','PAN12_DEV_POS':'100','PAN12_DEV_NEG':'500'})
import sys,json,hashlib
import numpy as np,torch
import torch.nn.functional as F
from torch import nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer,AutoModel
from pan12_pilot_data import ROOT,rows,y,dev,render
runtime=os.environ.get('PHOC_RUNTIME')
if not runtime:raise SystemExit('Set PHOC_RUNTIME to the pinned runtime checkout')
sys.path.insert(0,runtime)
from phocinae.engine import Engine,QTYPES
from phocinae.tokenizer import GemmaTokenizer
p=ROOT/'reports/pan12-matched-selection.json'
if not p.exists():raise SystemExit('No selection; train before verification')
x=json.loads(p.read_text());name=x['selected'];chosen=x['models'][name]
weights=ROOT/chosen['artifact'];assert hashlib.sha256(weights.read_bytes()).hexdigest()==chosen['sha256']
pdir=snapshot_download('Phocinae/Phocinae-Largha-150M-v1',revision='a57030322df13c8c55ab7a99b8b7b176e2c5aa85',local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
tok=AutoTokenizer.from_pretrained('google/mobilebert-uncased',revision='1f90a6c24c7879273a291d34a849033eba2dbc0f',local_files_only=True)
e=Engine.__new__(Engine);e.tok=GemmaTokenizer(pdir+'/tokenizer');e._head_cache={};e._opt_cache={}
q={'id':'risk','type':'noul','instructions':'Does this conversation contain grooming of a minor?'};spec=e._question_spec(q)
room=128-len(e._build_sequence(spec,[],[0,1],64,128)[0]);assert room==x['context_budget_phocinae']
def prefix(t):
 t=t.replace(e.tok.mask_token,' ')
 def fits(v):return len(e.tok.encode(v))<=room and len(tok(v,add_special_tokens=True)['input_ids'])<=128
 if fits(t):return t
 lo,hi=0,len(t)
 while lo<hi:
  mid=(lo+hi+1)//2
  if fits(t[:mid]):lo=mid
  else:hi=mid-1
 t=t[:lo]
 while t and not fits(t):t=t[:-1]
 return t
torch.set_num_threads(4)
if name=='mobilebert':
 class Mobile(nn.Module):
  def __init__(self):
   super().__init__();self.encoder=AutoModel.from_pretrained('google/mobilebert-uncased',revision='1f90a6c24c7879273a291d34a849033eba2dbc0f',local_files_only=True);self.head=nn.Linear(self.encoder.config.hidden_size,2)
  def forward(self,b):return self.head(F.layer_norm(self.encoder(**b).pooler_output,(self.encoder.config.hidden_size,)))
 model=Mobile().eval()
elif name=='phocinae':
 e=Engine(pdir,device='cpu',warm=False,threads=4);model=e.model.eval()
else:raise RuntimeError('Unexpected model selected')
loaded=model.load_state_dict(load_file(str(weights)),strict=False);assert not loaded.unexpected_keys
scores=[]
with torch.inference_mode():
 for off in range(0,len(dev),8):
  texts=[prefix(render(rows[i][1],10)) for i in dev[off:off+8]]
  if name=='mobilebert':logits=model(tok(texts,padding=True,truncation=True,max_length=128,return_tensors='pt'))
  else:
   items=[]
   for t in texts:
    ids,markers=e._build_sequence(spec,e.tok.encode(t),[0,1],64,128)
    items.append({'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':q})
   b=e._collate(items);logits=model(b['input_ids'],b['attention_mask'],b['marker_pos'],b['marker_mask'],b['qtype'])[0]
  scores.extend(torch.softmax(logits.float(),dim=1)[:,1].tolist())
tn,fp,fn,tp=confusion_matrix(y[dev],np.asarray(scores)>=chosen['threshold'],labels=[0,1]).ravel()
d=chosen['development'];assert (tp,fp,tn,fn)==(d['tp'],d['fp'],d['tn'],d['fn']),'Restored model differs from selection'
report={'selection_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'result':'selected model restored and development confusion matrix reproduced','official_test_opened':False}
(ROOT/'reports/pan12-matched-validated.json').write_text(json.dumps(report,indent=2)+'\n');print(report['result'])
