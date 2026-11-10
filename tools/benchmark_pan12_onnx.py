#!/usr/bin/env python3
"""Host-only synthetic-input ONNX benchmark; not a substitute for phone tests."""
import json, platform, resource, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
root=Path(__file__).resolve().parents[1]
assets=root/'artifacts/pan12-mobilebert-pilot'
tok=AutoTokenizer.from_pretrained(assets,local_files_only=True)
report={'platform':platform.platform(),'architecture':platform.machine(),'provider':'CPUExecutionProvider','threads':4,
        'scope':'Host-only, synthetic inputs; not Android latency, battery, or real workloads','models':{}}
for name in ['model.onnx','model-int8.onnx','model-uint8.onnx','model-int8-per-channel.onnx']:
    opts=ort.SessionOptions(); opts.intra_op_num_threads=4; opts.inter_op_num_threads=1
    start=time.perf_counter(); session=ort.InferenceSession(str(assets/name),opts,providers=['CPUExecutionProvider']); load=time.perf_counter()-start
    inputs={x.name for x in session.get_inputs()}; sizes={}
    for desired in [16,64,128]:
        text='hello world conversation message ' * 100
        batch=tok(text,return_tensors='np',truncation=True,max_length=desired)
        feed={k:v.astype(np.int64) for k,v in batch.items() if k in inputs}
        for _ in range(3): session.run(None,feed)
        samples=[]
        for _ in range(20):
            start=time.perf_counter(); session.run(None,feed); samples.append((time.perf_counter()-start)*1000)
        sizes[str(desired)]={'actual_tokens':int(batch['input_ids'].shape[1]),'p50_ms':float(np.percentile(samples,50)),
                              'p95_ms':float(np.percentile(samples,95)),'runs':20}
    report['models'][name]={'bytes':(assets/name).stat().st_size,'session_load_ms':load*1000,'lengths':sizes,
        'process_peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'rss_note':'Cumulative process peak; not per-model memory'}
    del session
(root/'reports').mkdir(exist_ok=True)
(root/'reports/pan12-host-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
