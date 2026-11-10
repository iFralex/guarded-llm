#!/usr/bin/env python3
"""Seven model probes on the same short, already-explored CYBY24 subset."""
import argparse,hashlib,json,os,random,re,sys
from collections import Counter
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel,AutoModelForSequenceClassification,AutoTokenizer,BertTokenizerFast
from safetensors.torch import load_file
from cyby24_split import ROOT,load
from cyby24_common_input import build

p=argparse.ArgumentParser();p.add_argument('--model',choices=['mobilebert','modernbert','phocinae','roberta','toxic','vigil','gemma'],required=True);p.add_argument('--verify-development',action='store_true');args=p.parse_args()
name=args.model
if args.verify_development and name not in ('mobilebert','modernbert','phocinae','roberta'):p.error('development verification is for the matched models')
dest=ROOT/(f'docs/cyby24-matched-verified-{name}.json' if args.verify_development else f'docs/cyby24-matched60-{name}.json')
if dest.exists():raise SystemExit('Existing report; refusing to overwrite')
torch.set_num_threads(4)
c,y,tr,dev,test,components=load();rng=random.Random(43)
sample=rng.sample([int(i) for i in test if y[i]==1],30)+rng.sample([int(i) for i in test if y[i]==0],30);rng.shuffle(sample)
if args.verify_development:sample=[int(i) for i in dev]
texts,cut,room=build(c,sample)
print('prepared',len(sample),'prefixes, truncated',sum(cut.values()),flush=True)
rev='1f90a6c24c7879273a291d34a849033eba2dbc0f'; mid='google/mobilebert-uncased'; pathmodern=ROOT/'.cache/modernbert-base'; pathroberta=ROOT/'.cache/roberta-cyberbullying'
threshold=None;model=None;engine=None;tok=None
class EncoderClassifier(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='mobilebert':self.encoder=AutoModel.from_pretrained(mid,revision=rev,local_files_only=True)
        else:self.encoder=AutoModel.from_pretrained(pathmodern,local_files_only=True,attn_implementation='eager')
        self.head=nn.Linear(self.encoder.config.hidden_size,2)
    def forward(self,b):
        h=self.encoder(**b);z=h.pooler_output if self.kind=='mobilebert' else h.last_hidden_state[:,0,:]
        return self.head(F.layer_norm(z,(self.encoder.config.hidden_size,)))
if name in ('mobilebert','modernbert','phocinae','roberta'):
    previous=json.loads((ROOT/f'reports/cyby24-matched-{name}.json').read_text());threshold=previous['threshold']
    if name=='roberta':
        model=AutoModelForSequenceClassification.from_pretrained(pathroberta,local_files_only=True);tok=AutoTokenizer.from_pretrained(pathroberta,local_files_only=True)
    elif name in ('mobilebert','modernbert'):
        model=EncoderClassifier(name);tok=AutoTokenizer.from_pretrained(mid,revision=rev,local_files_only=True) if name=='mobilebert' else AutoTokenizer.from_pretrained(pathmodern,local_files_only=True)
    else:
        from huggingface_hub import snapshot_download
        from phocinae.engine import Engine,QTYPES
        pdir=snapshot_download('Phocinae/Phocinae-Largha-150M-v1',revision='a57030322df13c8c55ab7a99b8b7b176e2c5aa85',local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
        engine=Engine(pdir,device='cpu',warm=False,threads=4);model=engine.model
        question={'id':'risk','type':'noul','instructions':'Does this conversation contain cyberbullying?'};spec=engine._question_spec(question)
    if name!='roberta':
        artifact=ROOT/previous['artifact']['path'];digest=hashlib.sha256(artifact.read_bytes()).hexdigest();assert digest==previous['artifact']['sha256']
        loaded=model.load_state_dict(load_file(str(artifact)),strict=False);assert not loaded.unexpected_keys
    model.eval()
elif name=='toxic':
    import onnxruntime as ort
    assets=ROOT/'app/src/main/assets/model';tok=BertTokenizerFast(vocab_file=str(assets/'vocab.txt'),do_lower_case=True)
    model=ort.InferenceSession(str(assets/'toxic-bert.onnx'),providers=['CPUExecutionProvider']); inputs={x.name for x in model.get_inputs()};threshold=.5
elif name=='vigil':
    from safecircle_probe import vigilant,file_sha,HASHES
    artifact='/Volumes/ORICO/guarded-llm-models/vigil-f16.gguf';assert file_sha(artifact)==HASHES['vigil']
    cli=ROOT/'.cache/vigil-llama/build-arm64/bin/llama-laya-cli'
else:
    import litert_lm as lm
    artifact='/Volumes/ORICO/guarded-llm-models/gemma-4-E2B-it.litertlm'
    h=hashlib.sha256()
    with open(artifact,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()=='181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c'
    system=json.loads((ROOT/'docs/gemma4-cyby24-probe.json').read_text())['system_prompt']
    engine=lm.Engine(artifact,backend=lm.Backend.CPU(thread_count=4),max_num_tokens=1024,cache_dir=None);engine.__enter__()
if args.verify_development:
    # Reproduce the original scorer's batch size and padding, since a
    # threshold chosen from an exact dev score can flip under roundoff.
    counts=Counter()
    with torch.inference_mode():
        for off in range(0,len(sample),4):
            group=sample[off:off+4]
            if name=='phocinae':
                items=[]
                for i in group:
                    ids,markers=engine._build_sequence(spec,engine.tok.encode(texts[i]),[0,1],64,256)
                    items.append({'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':question})
                b=engine._collate(items)
                logits=model(b['input_ids'],b['attention_mask'],b['marker_pos'],b['marker_mask'],b['qtype'])[0]
            else:
                b=tok([texts[i] for i in group],padding=True,truncation=True,max_length=256,return_tensors='pt')
                logits=model(**b).logits if name=='roberta' else model(b)
            for i,score in zip(group,torch.softmax(logits.float(),dim=1)[:,1].tolist()):
                pred=bool(score>=threshold)
                counts[{(1,True):'tp',(1,False):'fn',(0,True):'fp',(0,False):'tn'}[(int(y[i]),pred)]]+=1
    expected={k:previous['development'][k] for k in ('tp','tn','fp','fn')}
    actual={k:counts[k] for k in ('tp','tn','fp','fn')}
    print('restored development counts',actual,'expected',expected,flush=True)
    if actual!=expected:raise RuntimeError('Restored development confusion matrix differs')
    out={'model':name,'sample':'complete development split, batch size 4, original padding','result':'restored model and threshold reproduced development confusion counts','counts':actual,'truncated':int(sum(cut.values())),'limitations':'Restore check only, not independent evaluation.'}
    dest.write_text(json.dumps(out,indent=2)+'\n');sys.exit(0)
counts=Counter()
try:
    for j,i in enumerate(sample):
        text=texts[i];label=int(y[i]);pred=None;kind='invalid'
        try:
            if name in ('mobilebert','modernbert','roberta'):
                b=tok(text,truncation=True,max_length=256,return_tensors='pt')
                with torch.inference_mode(): logits=model(**b).logits if name=='roberta' else model(b)
                score=torch.softmax(logits.float(),dim=1)[0,1].item();pred=bool(score>=threshold)
            elif name=='phocinae':
                ids,markers=engine._build_sequence(spec,engine.tok.encode(text),[0,1],64,256)
                item={'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':question};b=engine._collate([item])
                with torch.inference_mode():logits=model(b['input_ids'],b['attention_mask'],b['marker_pos'],b['marker_mask'],b['qtype'])[0]
                score=torch.softmax(logits.float(),dim=1)[0,1].item();pred=bool(score>=threshold)
            elif name=='toxic':
                b=tok(text,truncation=True,max_length=256,return_tensors='np');logit=model.run(None,{k:v.astype(np.int64) for k,v in b.items() if k in inputs})[0][0][0]
                pred=bool(float(1/(1+np.exp(-logit)))>=threshold)
            elif name=='vigil':pred='bullying' in vigilant(cli,artifact,text)['flagged']
            else:
                with engine.create_conversation(system_message=system,sampler_config=lm.SamplerConfig(top_k=1,temperature=0.0,seed=42),thinking_config=lm.ThinkingConfig(enable_thinking=False,thinking_token_budget=0),max_output_tokens=16) as conv:
                    response=str(conv.send_message('THREAD START\n'+text+'\nTHREAD END',max_output_tokens=16)).strip()
                match=re.fullmatch(r'\s*(BULLYING|NOT_BULLYING|UNCERTAIN)[.!\s]*',response,re.I)
                if match:
                    ans=match.group(1).upper();pred={'BULLYING':True,'NOT_BULLYING':False}.get(ans);kind='uncertain'
        except Exception:pred=None;kind='error'
        if pred is None:
            counts['abstain_positive' if label else 'abstain_negative']+=1;counts[kind]+=1
        else:counts[{(1,True):'tp',(1,False):'fn',(0,True):'fp',(0,False):'tn'}[(label,pred)]]+=1
        if (j+1)%20==0:print('evaluated',j+1,'/',len(sample),flush=True)
finally:
    if name=='gemma' and engine:engine.close()
out={'model':name,'sample':'full development split for independent reload validation' if args.verify_development else 'same prior seed-43 balanced 60 CYBY24 threads, 30 positive and 30 negative','input':'identical raw prefix <=256 tokens for four tokenizers (including Phocinae question); 223 Phocinae context tokens','truncated':int(sum(cut.values())),'threshold':threshold,'threshold_policy':'mobilebert, modernbert, phocinae and roberta: threshold selected on full development for >=0.8 recall and max precision; toxic: fixed .5; Vigil: publisher category threshold; Gemma: fixed three-way bullying prompt', 'counts':dict(counts),'limitations':'Repeated previously explored, single-source sample; not independent or representative of Instagram. Different model training, thresholds and task definitions; Vigil/Gemma token limits not independently audited. No source text or IDs stored; no Android timing.'}
dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
