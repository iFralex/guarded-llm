#!/usr/bin/env python3
"""Exploratory matched-prefix pilot; stores aggregate metrics, never source texts."""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from transformers import BertTokenizer
from benchmark_metrics import calibration_threshold, summarize

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path('/Volumes/ORICO/guarded-llm-models')


def grouped_sample(indices, labels, seed):
    rng = random.Random(seed)
    positive = [int(i) for i in indices if labels[i] == 1]
    negative = [int(i) for i in indices if labels[i] == 0]
    chosen = rng.sample(positive, 20) + rng.sample(negative, 20)
    rng.shuffle(chosen)
    return chosen


def data(dataset):
    if dataset.startswith('synthetic_'):
        task = dataset.removeprefix('synthetic_')
        cases = json.loads((ROOT/'data/synthetic-new-pilot.json').read_text())['cases']
        cases = [c for split in ('calibration', 'evaluation') for c in cases if c['task'] == task and c['split'] == split]
        return [c['chat'] for c in cases], [int(c['label']) for c in cases]
    if dataset == 'grooming':
        os.environ.update({'PAN12_TRAIN_POS':'400','PAN12_TRAIN_NEG':'800','PAN12_CAL_POS':'100','PAN12_CAL_NEG':'500','PAN12_DEV_POS':'100','PAN12_DEV_NEG':'500'})
        from pan12_pilot_data import rows, cal, dev, render
        labels = [r[2] for r in rows]
        chosen = grouped_sample(cal, labels, 230) + grouped_sample(dev, labels, 231)
        texts = [render(rows[i][1], 10) for i in chosen]
    else:
        from cyby24_split import load
        conversations, labels, _, cal, test, _ = load()
        chosen = grouped_sample(cal, labels, 230) + grouped_sample(test, labels, 231)
        texts = [' '.join(f'Message {j+1}: {v}' for j,v in enumerate(conversations[i])) for i in chosen]
    return texts, [int(labels[i]) for i in chosen]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['grooming','bullying','synthetic_grooming','synthetic_bullying'], required=True)
    p.add_argument('--model', choices=['companion','espd_bert','espd_mobile','phocinae','llamaguard','toxic','roberta'], required=True)
    p.add_argument('--budget', type=int, default=128)
    p.add_argument('--run-tag', default='')
    args = p.parse_args()
    if not 64 <= args.budget <= 384:
        p.error('Budget must be between 64 and 384 tokens')
    if args.run_tag and not re.fullmatch(r'[a-z0-9_-]+', args.run_tag):
        p.error('Invalid run tag')
    suffix = (f'-{args.budget}' if args.budget != 128 or args.run_tag else '') + (f'-{args.run_tag}' if args.run_tag else '')
    dest = ROOT / 'reports' / f'matched-pilot-{args.dataset}-{args.model}{suffix}.json'
    if dest.exists():
        p.error('Output already exists')
    texts, labels = data(args.dataset)
    from huggingface_hub import snapshot_download
    sys.path.insert(0, os.environ['PHOC_RUNTIME'])
    from phocinae.engine import Engine, QTYPES
    from phocinae.tokenizer import GemmaTokenizer
    pdir = snapshot_download('Phocinae/Phocinae-Largha-150M-v1', revision='a57030322df13c8c55ab7a99b8b7b176e2c5aa85', local_files_only=True, allow_patterns=['model.safetensors','rl_agent_config.json','encoder/config.json','tokenizer/*'])
    tok = BertTokenizer(vocab_file=str(ROOT/'app/src/main/assets/model/vocab.txt'), do_lower_case=True)
    from transformers import AutoTokenizer
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROOT/'.cache/roberta-cyberbullying', local_files_only=True)
    builder = Engine.__new__(Engine)
    builder.tok = GemmaTokenizer(pdir+'/tokenizer')
    builder._head_cache = {}; builder._opt_cache = {}
    is_grooming = args.dataset.endswith('grooming')
    question = {'id':'risk','type':'noul','instructions': 'Does this conversation contain grooming of a minor?' if is_grooming else 'Does this conversation contain cyberbullying?'}
    spec = builder._question_spec(question)
    room = args.budget-len(builder._build_sequence(spec, [], [0,1], 64, args.budget)[0])
    def fits(text):
        return (len(builder.tok.encode(text)) <= room
                and len(tok(text, add_special_tokens=True)['input_ids']) <= args.budget
                and len(roberta_tokenizer(text, add_special_tokens=True)['input_ids']) <= args.budget)
    truncated = 0
    for j, text in enumerate(texts):
        text = text.replace(builder.tok.mask_token, ' ')
        if not fits(text):
            truncated += 1
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo+hi+1)//2
                if fits(text[:mid]): lo = mid
                else: hi = mid-1
            text = text[:lo]
            while text and not fits(text): text=text[:-1]
        texts[j] = text
    model = args.model
    scores = []
    if model in ('companion','espd_bert','espd_mobile'):
        os.environ['TF_USE_LEGACY_KERAS']='1'
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4)
        if model == 'companion':
            from transformers import BertConfig, TFBertModel
            encoder = TFBertModel(BertConfig())
            ids = tf.keras.layers.Input(shape=(400,),dtype=tf.int64,name='input_ids')
            types = tf.keras.layers.Input(shape=(400,),dtype=tf.int64,name='token_type_ids')
            mask = tf.keras.layers.Input(shape=(400,),dtype=tf.int64,name='attention_mask')
            sequence = encoder({'input_ids':ids,'token_type_ids':types,'attention_mask':mask})[0]
            features = [tf.keras.layers.GlobalMaxPooling1D()(tf.keras.layers.Conv1D(f,w,activation='relu')(sequence)) for w,f in ((3,100),(4,100),(5,50),(10,25))]
            h=tf.keras.layers.concatenate(features,axis=1)
            h=tf.keras.layers.Dropout(.1)(tf.keras.layers.Dense(200,activation='relu',name='hidden_layer')(h))
            h=tf.keras.layers.Dropout(.1)(tf.keras.layers.Dense(100,activation='relu',name='hidden_layer2')(h))
            net=tf.keras.Model(inputs=[ids,types,mask],outputs=tf.keras.layers.Dense(1,activation='sigmoid',name='classification_layer')(h))
            net.load_weights(str(EXTERNAL/'companion/bert_cnn_ensemble_resample_uncased_mdl.h5'))
        else:
            pattern = '*__mobilebert_classifier*PANC*' if model == 'espd_mobile' else '*__bert_classifier*PANC*'
            folder = next((EXTERNAL/'espd/demo').glob(pattern))
            tokenizer = BertTokenizer(vocab_file=str(folder/'vocab'),do_lower_case=True)
            interpreter = tf.lite.Interpreter(model_path=str(folder/'quantized/model.tflite'),num_threads=4)
            interpreter.allocate_tensors()
            inputs = {name:item['index'] for item in interpreter.get_input_details() for name in ('input_word_ids','input_mask','input_type_ids') if name in item['name']}
            output=interpreter.get_output_details()[0]['index']
        for text in texts:
            if model=='companion':
                b=tok(text,max_length=400,truncation=True,padding='max_length',return_tensors='tf')
                scores.append(float(net([b.input_ids,b.token_type_ids,b.attention_mask],training=False).numpy()[0][0]))
            else:
                b=tokenizer(text,max_length=512,truncation=True,padding='max_length',return_tensors='np')
                for name,key in (('input_word_ids','input_ids'),('input_mask','attention_mask'),('input_type_ids','token_type_ids')):
                    interpreter.set_tensor(inputs[name],b[key].astype(np.int32))
                interpreter.invoke(); scores.append(float(interpreter.get_tensor(output)[0][1]))
    elif model in ('toxic', 'roberta'):
        if model == 'toxic':
            import onnxruntime as ort
            net=ort.InferenceSession(str(ROOT/'app/src/main/assets/model/toxic-bert.onnx'),providers=['CPUExecutionProvider'])
            input_names={x.name for x in net.get_inputs()}
            for text in texts:
                batch=tok(text,max_length=args.budget,truncation=True,return_tensors='np')
                logits=net.run(None,{k:v.astype(np.int64) for k,v in batch.items() if k in input_names})[0][0][0]
                scores.append(float(1/(1+np.exp(-logits))))
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(4)
            folder=ROOT/'.cache/roberta-cyberbullying'
            tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
            net=AutoModelForSequenceClassification.from_pretrained(folder,local_files_only=True).eval()
            for text in texts:
                batch=tokenizer(text,max_length=args.budget,truncation=True,return_tensors='pt')
                with torch.inference_mode(): scores.append(float(torch.softmax(net(**batch).logits,dim=1)[0,1]))
    elif model=='phocinae':
        import torch
        from safetensors.torch import load_file
        torch.set_num_threads(4)
        engine=Engine(pdir,device='cpu',warm=False,threads=4)
        weights=ROOT/('artifacts/pan12-matched/phocinae.safetensors' if is_grooming else 'artifacts/cyby24-matched/phocinae.safetensors')
        engine.model.load_state_dict(load_file(str(weights)),strict=False); engine.model.eval()
        for text in texts:
            ids,markers=engine._build_sequence(spec,engine.tok.encode(text),[0,1],64,args.budget)
            item={'ids':ids,'markers':markers,'qtype':QTYPES['noul'],'n_opts':2,'spec':spec,'q':question}
            batch=engine._collate([item])
            with torch.inference_mode():
                logits=engine.model(batch['input_ids'],batch['attention_mask'],batch['marker_pos'],batch['marker_mask'],batch['qtype'])[0]
                scores.append(float(torch.softmax(logits.float(),dim=1)[0,1]))
    else:
        from llama_guard_synthetic_probe import PROMPT
        for text in texts:
            run=subprocess.run(['llama-cli','-m',str(EXTERNAL/'llama-guard/llama-guard-q3.gguf'),'-f','/dev/stdin','-n','32','-c','2048','-t','4','-ngl','0','--temp','0','--seed','42','--no-display-prompt','--no-warmup','--simple-io','--log-disable','--single-turn'],input=PROMPT.format(chat=text).encode('utf-8'),capture_output=True,timeout=180)
            if run.returncode: raise RuntimeError('Llama Guard inference failed')
            try: decoded=run.stdout.decode('utf-8')
            except UnicodeDecodeError:
                scores.append(None)
                continue
            match=re.search(r'(?im)^\s*(unsafe|safe)\s*$',decoded)
            scores.append(None if match is None else match.group(1).lower()=='unsafe')
    calibration_size = 8 if args.dataset.startswith('synthetic_') else 40
    if model=='llamaguard':
        decisions=scores; threshold=None
    else:
        if not all(np.isfinite(scores)):raise ValueError('Non-finite output')
        threshold=calibration_threshold(list(zip(labels[:calibration_size],scores[:calibration_size])),minimum_recall=.8)
        decisions=[s>=threshold for s in scores]
    out={'dataset':args.dataset,'model':model,'threshold':threshold,'calibration':summarize(list(zip(labels[:calibration_size],decisions[:calibration_size]))),
         'evaluation':summarize(list(zip(labels[calibration_size:],decisions[calibration_size:]))),'truncated':truncated,'room':room,'budget':args.budget,
         'limitations':'Synthetic cases are researcher-authored and labels are not independently reviewed. Real-data splits are previously explored; PAN12 labels are author-derived and overlap model training. Llama safety labels differ from target task.'}
    dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':
    main()
