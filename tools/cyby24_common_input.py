"""Reconstruct matched-prefix text in memory for selected CYBY24 indices."""
import os,sys
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
def build(conversations,indices):
    runtime=os.environ.get('PHOC_RUNTIME')
    if not runtime:raise RuntimeError('Set PHOC_RUNTIME')
    sys.path.insert(0,runtime)
    from phocinae.engine import Engine
    from phocinae.tokenizer import GemmaTokenizer
    repo='Phocinae/Phocinae-Largha-150M-v1';rev='a57030322df13c8c55ab7a99b8b7b176e2c5aa85'
    pdir=snapshot_download(repo,revision=rev,local_files_only=True,allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
    mt=AutoTokenizer.from_pretrained('google/mobilebert-uncased',revision='1f90a6c24c7879273a291d34a849033eba2dbc0f',local_files_only=True)
    nt=AutoTokenizer.from_pretrained(ROOT/'.cache/modernbert-base',local_files_only=True)
    rt=AutoTokenizer.from_pretrained(ROOT/'.cache/roberta-cyberbullying',local_files_only=True)
    e=Engine.__new__(Engine);e.tok=GemmaTokenizer(pdir+'/tokenizer');e._head_cache={};e._opt_cache={}
    q={'id':'risk','type':'noul','instructions':'Does this conversation contain cyberbullying?'};spec=e._question_spec(q); budget=256
    room=budget-len(e._build_sequence(spec,[],[0,1],64,budget)[0])
    def prefix(text):
        text=text.replace(e.tok.mask_token,' ')
        def fits(s):return len(e.tok.encode(s))<=room and all(len(t(s,add_special_tokens=True)['input_ids'])<=budget for t in (mt,nt,rt))
        if fits(text):return text,False
        lo,hi=0,len(text)
        while lo<hi:
            mid=(lo+hi+1)//2
            if fits(text[:mid]):lo=mid
            else:hi=mid-1
        while lo and not fits(text[:lo]):lo-=1
        return text[:lo],True
    texts={};cut={}
    for i in indices:
        original=' '.join(f'Message {j+1}: {v}' for j,v in enumerate(conversations[i]))
        texts[i],cut[i]=prefix(original)
    return texts,cut,room
