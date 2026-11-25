#!/usr/bin/env python3
"""Audit synthetic conversations against locally available prior messages, without storing source text."""
import json
import re
from collections import defaultdict
from pathlib import Path

from cyby24_split import load
from pan12_pilot_data import rows, render

ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def grams(text):
    words = re.findall(r'\w+', normalized(text))
    return set(zip(words, words[1:], words[2:], words[3:], words[4:]))


def main():
    cases = json.loads((ROOT/'data/synthetic-new-pilot.json').read_text())['cases']
    ids = [c['id'] for c in cases]
    assert len(ids) == len(set(ids))
    assert all(c['task'] in ('bullying','grooming') and c['split'] in ('calibration','evaluation') and c['label'] in (0,1) for c in cases)
    for task in ('bullying','grooming'):
        for split, expected in (('calibration',4),('evaluation',8)):
            part=[c for c in cases if c['task']==task and c['split']==split]
            assert sum(c['label'] for c in part)==expected and len(part)==2*expected
    conversations, _, _, _, _, _ = load()
    references = [render(row[1],10) for row in rows]
    references += [' '.join(f'Message {j+1}: {text}' for j,text in enumerate(thread)) for thread in conversations]
    for path in (ROOT/'data/gemma4-prompt-holdout.json', ROOT/'docs/synthetic-chat-probe.json'):
        prior=json.loads(path.read_text())
        references.extend(c['chat'] for c in prior['cases'] if 'chat' in c)
    exact = {normalized(x) for x in references}
    index = defaultdict(set)
    for i,text in enumerate(references):
        for gram in grams(text):
            index[gram].add(i)
    results=[]
    refgrams = {}
    for case in cases:
        g=grams(case['chat'])
        candidates=set()
        for gram in g:
            candidates.update(index.get(gram,()))
        maximum=0.0
        for i in candidates:
            if i not in refgrams:
                refgrams[i]=grams(references[i])
            other=refgrams[i]
            maximum=max(maximum,len(g & other)/len(g | other))
        results.append({'id':case['id'],'exact_match':normalized(case['chat']) in exact,
                        'max_fivegram_jaccard':round(maximum,4)})
    within=[]
    for i,a in enumerate(cases):
        for b in cases[i+1:]:
            ga,gb=grams(a['chat']),grams(b['chat'])
            sim=len(ga & gb)/len(ga | gb) if ga or gb else 0
            if sim >= .5:
                within.append({'a':a['id'],'b':b['id'],'fivegram_jaccard':round(sim,4)})
    report={'source_count':len(references),'cases':results,'within_synthetic_pairs_at_0_5':within,
            'limitations':'Only locally available selected PAN12 rows and CYBY24 filtered threads and two earlier synthetic sets; unknown model pretraining, paraphrases, authors and other datasets remain unaudited.'}
    (ROOT/'docs/synthetic-overlap-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('cases',len(cases),'exact matches',sum(x['exact_match'] for x in results),
          'max five-gram Jaccard',max(x['max_fivegram_jaccard'] for x in results),
          'within pairs >=0.5',len(within))


if __name__=='__main__':
    main()
