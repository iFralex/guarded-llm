"""Load a fixed, author-consensus sample from ConvAbuse without exporting text."""
import csv,hashlib,random,re
from pathlib import Path
import numpy as np
from cyby24_split import load as load_cyby24
ROOT=Path(__file__).resolve().parents[1]
def load():
    path=ROOT/'data/external/convabuse-test.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='8387d70332cd2245a095240534629152dd5729e8e9556e91b6b20dabdc0bb9d0'
    cyby,_,_,_,_,_=load_cyby24()
    norm=lambda s:re.sub(r'\s+',' ',s).strip().casefold()
    seen={norm(text) for thread in cyby for text in thread}
    pools={0:[],1:[]};excluded={'ambiguous_disagreement_or_insufficient_votes':0,'empty_or_exact_overlap':0}
    with path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            safe=bad=ambiguous=0
            for j in range(1,9):
                safe+=row.get(f'Annotator{j}_is_abuse.1')=='1'
                ambiguous+=row.get(f'Annotator{j}_is_abuse.0')=='1'
                bad+=any(row.get(f'Annotator{j}_is_abuse.{k}')=='1' for k in (-1,-2,-3))
            label=0 if safe>=2 and not bad and not ambiguous else 1 if bad>=2 and not safe and not ambiguous else None
            if label is None:excluded['ambiguous_disagreement_or_insufficient_votes']+=1;continue
            texts=[row.get(k,'').strip() for k in ('prev_agent','prev_user','agent','user') if row.get(k,'').strip()]
            if not row.get('user','').strip() or any(norm(t) in seen for t in texts):excluded['empty_or_exact_overlap']+=1;continue
            pools[label].append((row['conv_id'],texts))
    rng=random.Random(97);selected=[];used=set()
    for label in (1,0):
        rng.shuffle(pools[label]);n=0
        for conv,texts in pools[label]:
            if conv in used:continue
            selected.append((texts,label));used.add(conv);n+=1
            if n==50:break
        if n!=50:raise RuntimeError('Not enough unique conversations in a class')
    rng.shuffle(selected)
    return [x[0] for x in selected],np.array([x[1] for x in selected]),excluded
