#!/usr/bin/env python3
"""One-shot official PAN12 test after frozen development selection; emits aggregates only."""
import os
os.environ.update({'PAN12_TRAIN_POS':'400','PAN12_TRAIN_NEG':'800','PAN12_CAL_POS':'100','PAN12_CAL_NEG':'500','PAN12_DEV_POS':'100','PAN12_DEV_NEG':'500'})
import sys,io,json,hashlib,random,zipfile,re,xml.etree.ElementTree as ET
from collections import defaultdict,Counter
import numpy as np,torch
import torch.nn.functional as F
from torch import nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,average_precision_score
from transformers import AutoTokenizer,AutoModel
from pan12_pilot_data import ROOT,ARCHIVE,DIGEST,rows,tr,cal,dev,render

runtime=os.environ.get('PHOC_RUNTIME')
if not runtime:raise SystemExit('Set PHOC_RUNTIME to the pinned runtime checkout')
sys.path.insert(0,runtime)
from phocinae.engine import Engine,QTYPES
from phocinae.tokenizer import GemmaTokenizer
selection_path=ROOT/'reports/pan12-matched-selection.json'
out=ROOT/'reports/pan12-matched-test.json'
if out.exists():raise SystemExit('Official test already evaluated; refusing a second run.')
if not selection_path.exists():raise SystemExit('Development selection missing; no test access.')
selection=json.loads(selection_path.read_text())
assert selection['selection_locked_before_test'] and selection['selected'] in ('mobilebert','phocinae')
selection_sha=hashlib.sha256(selection_path.read_bytes()).hexdigest()
validation_path=ROOT/'reports/pan12-matched-validated.json'
if not validation_path.exists():raise SystemExit('Selected model has not been independently restored and validated')
validation=json.loads(validation_path.read_text())
if validation.get('selection_sha256')!=selection_sha or validation.get('official_test_opened') is not False:
    raise SystemExit('Validation does not match locked selection')
name=selection['selected'];chosen=selection['models'][name];threshold=chosen['threshold']
weights_path=ROOT/chosen['artifact'];assert hashlib.sha256(weights_path.read_bytes()).hexdigest()==chosen['sha256']
repo='Phocinae/Phocinae-Largha-150M-v1';rev='a57030322df13c8c55ab7a99b8b7b176e2c5aa85'
pdir=snapshot_download(repo,revision=rev,local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
tok=AutoTokenizer.from_pretrained('google/mobilebert-uncased',revision='1f90a6c24c7879273a291d34a849033eba2dbc0f',local_files_only=True)
engine=Engine.__new__(Engine);engine.tok=GemmaTokenizer(pdir+'/tokenizer');engine._head_cache={};engine._opt_cache={}
q={'id':'risk','type':'noul','instructions':'Does this conversation contain grooming of a minor?'};spec=engine._question_spec(q)
room=128-len(engine._build_sequence(spec,[],[0,1],64,128)[0]);assert room==selection['context_budget_phocinae']
def common_prefix(text):
    text=text.replace(engine.tok.mask_token,' ')
    def fits(s):return len(engine.tok.encode(s))<=room and len(tok(s,add_special_tokens=True)['input_ids'])<=128
    if fits(text):return text,False
    lo,hi=0,len(text)
    while lo<hi:
        mid=(lo+hi+1)//2
        if fits(text[:mid]):lo=mid
        else:hi=mid-1
    prefix=text[:lo]
    while prefix and not fits(prefix):prefix=prefix[:-1]
    return prefix,True

# Collect all authors from the original training archive, including excluded segments.
known_authors=set();fingerprints={hashlib.sha256(render(r[1],10).casefold().encode()).digest() for r in rows}
with zipfile.ZipFile(ARCHIVE) as outer:
    trzip=next(x for x in outer.namelist() if 'training-corpus' in x and x.endswith('.zip'))
    with zipfile.ZipFile(io.BytesIO(outer.read(trzip))) as z:
        xml=next(x for x in z.namelist() if x.endswith('training-corpus-2012-05-01.xml'))
        with z.open(xml) as stream:
            for _,node in ET.iterparse(stream,events=('end',)):
                if node.tag=='conversation':
                    known_authors.update(m.findtext('author','').strip() for m in node.findall('message'))
                    node.clear()
known_authors.discard('')
# Test labels come from author ground truth. Sample uniformly, not by class.
reservoir=[];rng=random.Random(43);eligible=0;counts=Counter()
with zipfile.ZipFile(ARCHIVE) as outer:
    testzip=next(x for x in outer.namelist() if 'test-corpus' in x and x.endswith('.zip'))
    with zipfile.ZipFile(io.BytesIO(outer.read(testzip))) as z:
        truth=next(x for x in z.namelist() if x.endswith('groundtruth-problem1.txt'))
        positives=set(z.read(truth).decode('utf-8-sig').split())
        xml=next(x for x in z.namelist() if x.endswith('test-corpus-2012-05-17.xml'))
        with z.open(xml) as stream:
            for _,node in ET.iterparse(stream,events=('end',)):
                if node.tag!='conversation':continue
                counts['segments']+=1; messages=node.findall('message')
                authors={m.findtext('author','').strip() for m in messages}-{''}
                if not authors:counts['no_author']+=1;node.clear();continue
                if authors & known_authors:counts['overlap_author']+=1;node.clear();continue
                positive=bool(authors & positives)
                early={m.findtext('author','').strip() for m in messages[:5]}-{''}
                if positive and not (early & positives):counts['positive_not_in_first_five']+=1;node.clear();continue
                content=[]
                for m in messages[:10]:
                    element=m.find('text');txt=''.join(element.itertext()) if element is not None else ''
                    content.append((m.findtext('author','').strip(),re.sub(r'\s+',' ',txt).strip()[:512]))
                text=render(content,10)
                if not text:counts['empty']+=1;node.clear();continue
                if hashlib.sha256(text.casefold().encode()).digest() in fingerprints:counts['exact_overlap']+=1;node.clear();continue
                eligible+=1;sample=(text,int(positive))
                if len(reservoir)<3000:reservoir.append(sample)
                else:
                    j=rng.randrange(eligible)
                    if j<3000:reservoir[j]=sample
                node.clear()
counts['eligible_before_sampling']=eligible; counts['sampled_before_near_duplicate_audit']=len(reservoir)
# Partial near-duplicate audit against all selected training/calibration/development segments.
def grams(text):
    words=re.findall(r'\w+',text.lower());return set(zip(words,words[1:],words[2:]))
ref=[];inverted=defaultdict(set)
for i in list(tr)+list(cal)+list(dev):
    g=grams(render(rows[i][1],10));idx=len(ref);ref.append(g)
    if len(g)>=8:
        for token in g:inverted[token].add(idx)
kept=[]
for text,label in reservoir:
    g=grams(text);candidates=set()
    if len(g)<8:counts['short_not_near_duplicate_audited']+=1
    else:
        for token in g:candidates.update(inverted.get(token,()))
        if any(len(g & ref[j])/len(g | ref[j])>=.8 for j in candidates):counts['near_duplicate_removed']+=1;continue
    kept.append((text,label))
if not kept or not any(label for _,label in kept):raise RuntimeError('No positive examples remain; do not interpret this as performance')
print('Eligible and sampled test segments',eligible,len(kept),'positives',sum(x[1] for x in kept),flush=True)
# Load only the locked development-selected model, never fit anything on test.
torch.set_num_threads(4)
if name=='mobilebert':
    class Mobile(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=AutoModel.from_pretrained('google/mobilebert-uncased',revision='1f90a6c24c7879273a291d34a849033eba2dbc0f',local_files_only=True);self.head=nn.Linear(self.encoder.config.hidden_size,2)
        def forward(self,b):return self.head(F.layer_norm(self.encoder(**b).pooler_output,(self.encoder.config.hidden_size,)))
    model=Mobile().eval()
else:
    engine=Engine(pdir,device='cpu',warm=False,threads=4);model=engine.model.eval()
loaded=model.load_state_dict(load_file(str(weights_path)),strict=False);assert not loaded.unexpected_keys
scores=[];labels=[];truncated=0
with torch.inference_mode():
    for off in range(0,len(kept),8):
        batch=kept[off:off+8];texts=[]
        for value,label in batch:
            prefix,cut=common_prefix(value);texts.append(prefix);labels.append(label);truncated+=cut
        if name=='mobilebert': logits=model(tok(texts,padding=True,truncation=True,max_length=128,return_tensors='pt'))
        else:
            items=[]
            for value in texts:
                ids,markers=engine._build_sequence(spec,engine.tok.encode(value),[0,1],64,128)
                items.append({'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':q})
            b=engine._collate(items);logits=model(b['input_ids'],b['attention_mask'],b['marker_pos'],b['marker_mask'],b['qtype'])[0]
        scores.extend(torch.softmax(logits.float(),dim=1)[:,1].tolist())
labels=np.asarray(labels);pred=np.asarray(scores)>=threshold;tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
report={'selected':name,'selection_sha256_before_test':selection_sha,'threshold_frozen':threshold,'corpus':'official PAN12 test, uniformly sampled after author and exact-duplicate exclusions','audit_counts':dict(counts),'n':len(labels),'positive':int(sum(labels)),
        'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'precision':precision_score(labels,pred,zero_division=0),
        'recall':recall_score(labels,pred,zero_division=0),'f1':f1_score(labels,pred,zero_division=0),
        'average_precision':average_precision_score(labels,scores),'matched_prefix_truncated':int(truncated),
        'limitations':'One uniform sample after exclusions, not full test; author-derived weak labels, filter on first five messages, incomplete near-duplicate audit (especially short inputs), source/decoy artifacts, no independent platform and unknown pretraining overlap; not validated for real-world grooming.'}
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
