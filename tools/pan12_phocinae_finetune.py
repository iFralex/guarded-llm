#!/usr/bin/env python3
"""Limited Phocinae fine-tune: last two encoder layers and decision head on PAN12 pilot."""
import hashlib,json,os,random,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,average_precision_score,brier_score_loss
from pan12_pilot_data import ROOT,rows,y,tr,cal,dev,render

runtime=os.environ.get('PHOC_RUNTIME')
if not runtime: raise SystemExit('Set PHOC_RUNTIME to the pinned phocinae-server checkout')
sys.path.insert(0,runtime)
from phocinae.engine import Engine,QTYPES
repo='Phocinae/Phocinae-Largha-150M-v1'; revision='a57030322df13c8c55ab7a99b8b7b176e2c5aa85'
path=snapshot_download(repo,revision=revision,local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
torch.set_num_threads(4); torch.manual_seed(42)
engine=Engine(path,device='cpu',warm=False,threads=4)
model=engine.model
# Keep the embedding table and early layers frozen to fit the CPU/memory budget.
for p in model.parameters(): p.requires_grad=False
for layer in list(model.encoder.layers)[-2:]:
    for p in layer.parameters(): p.requires_grad=True
for module in (model.encoder.final_norm,model.head,model.scorer,model.type_emb):
    for p in module.parameters(): p.requires_grad=True
# act_head does not participate in the supervised binary loss and remains frozen.
trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
optimizer=torch.optim.AdamW([
    {'params':[p for n,p in trainable.items() if n.startswith('encoder.')],'lr':2e-5},
    {'params':[p for n,p in trainable.items() if not n.startswith('encoder.')],'lr':1e-4},
],weight_decay=0.01)
question={'id':'risk','type':'noul','instructions':'Does this conversation contain grooming of a minor?'}
def item(i,window):
    context=render(rows[i][1],window)
    spec=engine._question_spec(question)
    state=engine.tok.encode(context.replace(engine.tok.mask_token,' '))
    ids,markers=engine._build_sequence(spec,state,[0,1],64,128)
    if len(markers)!=2: raise RuntimeError('Decision options exceed the model budget')
    return {'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':question}

def probabilities(indices,window):
    model.eval(); result=[]; start=time.perf_counter()
    with torch.inference_mode():
        for offset in range(0,len(indices),4):
            items=[item(i,window) for i in indices[offset:offset+4]]
            logits,act=engine._forward(engine._collate(items))
            result.extend(row[3][1] for row in engine._decode_rows(logits.numpy(),act.numpy(),items))
    print('evaluation',window,'examples',len(indices),'seconds',round(time.perf_counter()-start,2),flush=True)
    return np.asarray(result)

print('trainable parameters',sum(p.numel() for p in trainable.values()),'of',sum(p.numel() for p in model.parameters()),flush=True)
order=list(tr); rng=random.Random(42); epochs=[]
class_weights=torch.tensor([1.0,float((len(tr)-sum(y[tr]))/sum(y[tr]))],dtype=torch.float32)
for epoch in range(2):
    rng.shuffle(order); model.train(); model.encoder.eval()
    for layer in list(model.encoder.layers)[-2:]: layer.train()
    start=time.perf_counter(); total=0.0
    for off in range(0,len(order),4):
        ix=order[off:off+4]; batch=engine._collate([item(i,10) for i in ix])
        labels=torch.tensor(y[ix],dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits,_=model(batch['input_ids'],batch['attention_mask'],batch['marker_pos'],batch['marker_mask'],batch['qtype'])
        loss=F.cross_entropy(logits,labels,weight=class_weights)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0); optimizer.step()
        total+=loss.item()
    stats={'epoch':epoch+1,'mean_batch_loss':total/((len(order)+3)//4),'seconds':time.perf_counter()-start}
    epochs.append(stats); print('epoch',stats,flush=True)

calibration=probabilities(cal,10); cut=0.0
for value in sorted(set(calibration),reverse=True):
    if recall_score(y[cal],calibration>=value,zero_division=0)>=.9: cut=float(value); break
out={}; predictions={}
for window in (5,10,20):
    scores=probabilities(dev,window); predictions[window]=scores; pred=scores>=cut
    tn,fp,fn,tp=confusion_matrix(y[dev],pred,labels=[0,1]).ravel()
    out[str(window)]={'n':len(dev),'positives':int(sum(y[dev])),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
        'precision':precision_score(y[dev],pred,zero_division=0),'recall':recall_score(y[dev],pred,zero_division=0),
        'f1':f1_score(y[dev],pred,zero_division=0),'average_precision':average_precision_score(y[dev],scores),
        'brier_uncalibrated':brier_score_loss(y[dev],scores)}
# Export only trained parameter tensors; never conversation text. Base weights are required.
artifact=ROOT/'artifacts/pan12-phocinae-finetune'; artifact.mkdir(parents=True,exist_ok=True)
weights={n:p.detach().cpu().contiguous() for n,p in trainable.items()}
weight_path=artifact/'trainable.safetensors'
save_file(weights,str(weight_path),metadata={'base_model':repo,'base_revision':revision,'runtime_revision':'f751d7258e50b4cd19fe3967ae184ed895ce68b5'})
spec=engine._question_spec(question)
base=len(engine._build_sequence(spec,[],[0,1],64,128)[0]); room=128-base
truncation={}
for window in (5,10,20):
    lengths=[len(engine.tok.encode(render(rows[i][1],window).replace(engine.tok.mask_token,' '))) for i in dev]
    truncation[str(window)]={'n':len(lengths),'truncated':sum(v>room for v in lengths)}
report={'context_token_budget':room,'development_truncation':truncation,'model':repo,'revision':revision,'runtime_revision':'f751d7258e50b4cd19fe3967ae184ed895ce68b5',
 'mode':'fine-tune final 2 encoder layers, final norm, typed-decision head, scorer and type embeddings; other encoder layers and action head frozen',
 'input':'up to 5/10/20 messages, entire typed-decision sequence capped at 128 tokens including question; trained on first 10 messages',
 'partitions':{'train':{'n':len(tr),'positive':int(sum(y[tr]))},'calibration':{'n':len(cal),'positive':int(sum(y[cal]))},'development':{'n':len(dev),'positive':int(sum(y[dev]))}},
 'trainable_parameters':sum(p.numel() for p in trainable.values()),'total_parameters':sum(p.numel() for p in model.parameters()),
 'optimizer':'AdamW, encoder lr 2e-5, head lr 1e-4, weighted cross entropy, batch 4, seed 42','epochs':epochs,
 'threshold_policy':'highest sampled calibration score with recall >=0.9; no probability calibration','threshold':cut,'development':out,
 'artifact':{'path':'artifacts/pan12-phocinae-finetune/trainable.safetensors','bytes':weight_path.stat().st_size,'sha256':hashlib.sha256(weight_path.read_bytes()).hexdigest(),'contains':'trainable weights only, base checkpoint required'},
 'limitations':'Small sampled cohort with altered prevalence, weak author-derived labels, prompt-dependent typed decision, incomplete near-duplicate audit, source/decoy artifacts; tokenizer and prompt budget differ from MobileBERT. No official PAN12 test or phone benchmark.'}
(ROOT/'reports/pan12-phocinae-finetune.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
