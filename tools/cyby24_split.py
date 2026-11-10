"""Reconstruct the fixed, previously explored CYBY24 thread split in memory."""
import hashlib,io,re,zipfile
from pathlib import Path
import numpy as np, openpyxl
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/external/cyby24.zip'
seed=67

def load():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='fad01343cc62e58558101ff15d27bb484efe38ac1cfe31930aab55dd490a3be2'
    with zipfile.ZipFile(SOURCE) as z:
        sheet = openpyxl.load_workbook(io.BytesIO(z.read('CYBY24.xlsx')), read_only=True, data_only=True).active
    iterator = iter(sheet.values); headers = next(iterator)
    assert headers[5] == 'text' and headers[7] == 'cyberbullying_label'
    groups, current = [], []
    for row in iterator:
        if all(v is None or (isinstance(v,str) and not v.strip()) for v in row):
            if current: groups.append(current); current = []
        else: current.append(row)
    if current: groups.append(current)
    old_sheet = openpyxl.load_workbook(ROOT/'data/external/cyby23.xlsx', read_only=True, data_only=True).active
    old_rows = list(old_sheet.values)[1:]
    normalize = lambda text: re.sub(r'\s+', ' ', text).strip().casefold()
    old_texts = {normalize(r[5]) for r in old_rows if isinstance(r[5],str) and r[5].strip()}
    old_ids = {str(r[0]) for r in old_rows if r[0] is not None}
    conversations, y, ids = [], [], []
    excluded_old, excluded_invalid = 0, 0
    for group in groups:
        labels = {int(r[7]) for r in group if r[7] is not None and str(r[7]).strip()}
        texts = [r[5].strip() for r in group if isinstance(r[5], str) and r[5].strip()]
        if len(labels) != 1 or not texts or not labels <= {1,2,3,4}:
            excluded_invalid += 1; continue
        if any(normalize(t) in old_texts for t in texts) or any(str(r[0]) in old_ids for r in group if r[0] is not None):
            excluded_old += 1; continue
        conversations.append(texts); y.append(int(next(iter(labels)) in (3,4)))
        # Excel represents IDs as floats: conservative equality can over-group IDs.
        ids.append([str(r[0]) for r in group if r[0] is not None])
    y = np.array(y); parent = list(range(len(y)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    seen = {}
    for i,(texts,tweet_ids) in enumerate(zip(conversations,ids)):
        for key in [('text', re.sub(r'\s+',' ',text).strip().casefold()) for text in texts] + [('tweet_id', x) for x in tweet_ids]:
            if key in seen: union(i,seen[key])
            else: seen[key]=i
    components = np.array([find(i) for i in range(len(y))])
    # Fixed split before inference. Development and test are within this already explored corpus.
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)),y,components))
    test = folds[0][1]; dev = folds[1][1]; train = np.array(sorted(set(range(len(y)))-set(test)-set(dev)))
    assert not (set(components[train]) & set(components[dev]) or set(components[train]) & set(components[test]) or set(components[dev]) & set(components[test]))
    assert all(len(set(y[s])) == 2 for s in (train,dev,test))
    return conversations,y,train,dev,test,components
