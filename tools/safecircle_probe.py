#!/usr/bin/env python3
"""Run public SafeCircle research models locally on existing invented chats.

Vigil's fixed questions are loaded from the publisher's unmodified helper in
the ignored local cache. This wrapper uses stdin instead of the helper's
delete=False temporary file, so no conversation is intentionally persisted.
"""
import argparse, hashlib, importlib.util, json, re, subprocess, time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HASHES={'vigil':'6851d99ccc113738794ec1b4cc70baed1a15da80956bb77407c9e43f73b49068','horizon':'ce7580ad15a07c0bb6a70946d032532a0c3fc6c3112eea50b102b3e682a5fc33'}
REVISIONS={'vigil':'b239313708157db33314ecc81269e9babc48452a','horizon':'7fe3acf1e3a5338b613faa928c0321666f67e367'}
HORIZON_SYSTEM=('You are Horizon, SafeCircle\'s child safety risk detection model. You have no general knowledge or identity beyond this task. Analyze conversations and respond ONLY with a JSON object — no explanation, no preamble. JSON schema: {"risk_detected": bool, "category": "grooming|bullying|sexual_content|isolation|personal_info|platform_migration|threats|benign", "severity": "none|low|medium|high|critical", "confidence": 0.0-1.0, "reasoning": "one sentence max"}. If asked about yourself or anything unrelated to risk analysis, respond: {"error": "I only analyze conversations for child safety risks."}')
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def load_vigil():
    spec=importlib.util.spec_from_file_location('official_vigil',ROOT/'.cache/safecircle/run_vigil.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    thresholds=json.loads((ROOT/'.cache/safecircle/vigil_spec.json').read_text())['thresholds']
    return module.QUESTIONS,module.CATEGORIES,thresholds
def vigilant(cli,model,text):
    questions,categories,thresholds=load_vigil()
    req=json.dumps({'state':text,'questions':questions})
    proc=subprocess.run([str(cli),'-m',str(model),'-f','-','-t','4'],input=req,text=True,capture_output=True,timeout=120)
    if proc.returncode: raise RuntimeError('Vigil CLI failed (details omitted to avoid logging input)')
    answers=json.loads(proc.stdout)['answers']
    probabilities={k:answers['category']['probabilities'][k] for k in categories}
    return {'flagged':[k for k in categories if probabilities[k]>=thresholds[k]],'scores':probabilities}
def parse_horizon(answer):
    start=answer.find('{')
    if start<0: return None
    try: data,_=json.JSONDecoder().raw_decode(answer[start:])
    except (ValueError,TypeError): return None
    categories={'grooming','bullying','sexual_content','isolation','personal_info','platform_migration','threats','benign'}
    if not isinstance(data,dict) or type(data.get('risk_detected')) is not bool or data.get('category') not in categories: return None
    if data['risk_detected'] == (data['category']=='benign'): return None
    return data
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--kind',choices=list(HASHES),required=True); parser.add_argument('--model',required=True); parser.add_argument('--cli'); args=parser.parse_args()
    assert file_sha(args.model)==HASHES[args.kind]
    if args.kind=='vigil' and not args.cli: parser.error('--cli required for Vigil')
    old=json.loads((ROOT/'docs/synthetic-chat-probe.json').read_text())['cases']
    labels={'school_plans':0,'teen_friendship':0,'sarcastic_joke':0,'support':0,'bullying':1,'threat':1,'adult_secrecy':1,'request_private_images':1,'coach_secrecy':0}
    cases=[{'id':'prior_'+c['id'],'chat':c['chat'],'expected':labels.get(c['id'])} for c in old]
    cases += [{'id':'holdout_'+c['id'],'chat':c['chat'],'expected':int(c['label']=='CONCERN')} for c in json.loads((ROOT/'data/gemma4-prompt-holdout.json').read_text())['cases']]
    report={'kind':args.kind,'repo':'safecircleai/'+('vigil' if args.kind=='vigil' else 'horizon-mobile'),'revision':REVISIONS[args.kind],'model_sha256':HASHES[args.kind], 'license':'SafeCircle Research License SRL-1.0, noncommercial research', 'unit':'invented conversation; author-intended labels, not independent expert judgment','cases':[], 'limitations':'Previously examined examples, synthetic and not representative. Category tasks and inputs differ from prior models. No Android measurement; avoid using for deployment.'}
    if args.kind=='horizon': report['system_message']=HORIZON_SYSTEM
    engine=None
    if args.kind=='horizon':
        import litert_lm as lm
        engine=lm.Engine(args.model,backend=lm.Backend.CPU(thread_count=4),max_num_tokens=1280,cache_dir=None)
        engine.__enter__()
    try:
        for c in cases:
            start=time.perf_counter()
            try:
                if args.kind=='vigil':
                    output=vigilant(args.cli,args.model,c['chat'])
                    flag=bool(output['flagged']); raw=None
                else:
                    with engine.create_conversation(system_message=HORIZON_SYSTEM,sampler_config=lm.SamplerConfig(top_k=1,temperature=0.0,seed=42),max_output_tokens=160) as conversation:
                        raw=str(conversation.send_message('Analyze this conversation:\n'+c['chat'],max_output_tokens=160)).strip()
                    output=parse_horizon(raw); flag=output['risk_detected'] if output is not None else None
                item={**c,'flagged':flag,'result':output,'seconds':round(time.perf_counter()-start,3)}
                if args.kind=='horizon': item['raw_answer']=raw
            except Exception as ex: item={**c,'flagged':None,'error':type(ex).__name__,'seconds':round(time.perf_counter()-start,3)}
            report['cases'].append(item)
            print(c['id'],'flag',item['flagged'],'seconds',item['seconds'],flush=True)
            (ROOT/f'reports/{args.kind}-synthetic-partial.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        if engine: engine.close()
    counts=Counter()
    for c in report['cases']:
        if c['expected'] is None: counts['unscored']+=1
        elif c['flagged'] is None: counts['invalid_positive' if c['expected'] else 'invalid_negative']+=1
        else: counts[{(1,True):'tp',(1,False):'fn',(0,True):'fp',(0,False):'tn'}[(c['expected'],c['flagged'])]]+=1
    report['counts']=dict(counts)
    (ROOT/f'docs/{args.kind}-synthetic-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print('counts',dict(counts),flush=True)
if __name__=='__main__': main()
