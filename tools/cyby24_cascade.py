#!/usr/bin/env python3
"""Compare Toxic-BERT alone with Toxic-BERT followed by Vigil on CYBY24."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizerFast

from benchmark_metrics import calibration_threshold, summarize
from cyby24_common_input import build
from cyby24_split import ROOT, load
from safecircle_probe import HASHES, file_sha, load_vigil, vigilant


def report_partition(rows, gate, vigil_threshold):
    baseline = summarize([(label, score >= gate) for label, score, _ in rows])
    cascade = summarize([
        (label, vigil_score is not None and vigil_score >= vigil_threshold)
        for label, _, vigil_score in rows
    ])
    before, after = baseline['counts'], cascade['counts']
    forwarded = sum(vigil_score is not None for _, _, vigil_score in rows)
    return {
        'baseline': baseline,
        'cascade': cascade,
        'second_stage_calls': forwarded,
        'second_stage_call_rate': forwarded / len(rows),
        'positives_missed_at_gate': before['fn'],
        'positives_rejected_at_second_stage': before['tp'] - after['tp'],
        'false_alarms_removed_by_second_stage': before['fp'] - after['fp'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vigil-model', type=Path, required=True)
    parser.add_argument('--vigil-cli', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/cyby24-cascade-vigil.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output file already exists')
    if file_sha(args.vigil_model) != HASHES['vigil']:
        parser.error('Vigil checkpoint does not match the pinned SHA-256')
    _, _, thresholds = load_vigil()
    publisher_threshold = thresholds['bullying']

    reference = json.loads((ROOT / 'docs/cyby24-standardized.json').read_text())
    toxic_reference = next(r for r in reference['results'] if r['model'] == 'toxic')
    gate = toxic_reference['threshold']
    conversations, labels, _, dev, test, _ = load()
    sample = [int(i) for i in (*dev, *test)]
    assets = ROOT / 'app/src/main/assets/model'
    tokenizer = BertTokenizerFast(vocab_file=str(assets / 'vocab.txt'), do_lower_case=True)
    texts, truncated, _ = build(conversations, sample, extra_tokenizers=(tokenizer,))
    cuts = {
        'calibration': sum(truncated[int(i)] for i in dev),
        'evaluation': sum(truncated[int(i)] for i in test),
    }
    if cuts != toxic_reference['truncated']:
        raise RuntimeError('The common prefix no longer matches the reference run')

    session = ort.InferenceSession(str(assets / 'toxic-bert.onnx'), providers=['CPUExecutionProvider'])
    input_names = {value.name for value in session.get_inputs()}
    toxic_scores = []
    for offset, i in enumerate(sample, start=1):
        batch = tokenizer(texts[i], truncation=True, max_length=256, return_tensors='np')
        feed = {key: value.astype(np.int64) for key, value in batch.items() if key in input_names}
        logit = session.run(None, feed)[0][0][0]
        toxic_scores.append(float(1 / (1 + np.exp(-logit))))
        if offset % 100 == 0:
            print(f'Toxic-BERT scored {offset}/{len(sample)}', flush=True)

    for start, stop, split in ((0, len(dev), 'calibration'), (len(dev), len(sample), 'evaluation')):
        baseline = summarize([(int(labels[sample[j]]), toxic_scores[j] >= gate) for j in range(start, stop)])
        if baseline['counts'] != toxic_reference[split]['counts']:
            raise RuntimeError(f'Toxic-BERT baseline does not reproduce the {split} confusion matrix')

    rows = []
    forwarded = 0
    started = time.perf_counter()
    for i, toxic_score in zip(sample, toxic_scores, strict=True):
        vigil_score = None
        if toxic_score >= gate:
            try:
                vigil_score = vigilant(args.vigil_cli, args.vigil_model, texts[i])['scores']['bullying']
            except Exception:
                raise RuntimeError(f'Vigil inference failed at forwarded case {forwarded + 1}') from None
            if not np.isfinite(vigil_score):
                raise RuntimeError('Vigil returned a non-finite bullying score')
            forwarded += 1
            if forwarded % 25 == 0:
                print(f'Vigil reviewed {forwarded} forwarded threads', flush=True)
        rows.append((int(labels[i]), toxic_score, vigil_score))

    calibration, evaluation = rows[:len(dev)], rows[len(dev):]
    calibrated_threshold = calibration_threshold(
        [(label, vigil_score) for label, _, vigil_score in calibration if vigil_score is not None],
        minimum_recall=0.95,
    )
    policies = {}
    for name, threshold in (('publisher', publisher_threshold), ('calibrated', calibrated_threshold)):
        cal_results = report_partition(calibration, gate, threshold)
        target_recall = cal_results['cascade']['recall']
        # Compare a simpler strategy locked to the same minimum calibration recall.
        toxic_only_threshold = calibration_threshold(
            [(label, score) for label, score, _ in calibration], minimum_recall=target_recall,
        )
        eval_results = report_partition(evaluation, gate, threshold)
        policies[name] = {
            'vigil_threshold': threshold,
            'calibration': cal_results,
            'evaluation': eval_results,
            'toxic_only_recalibrated': {
                'threshold': toxic_only_threshold,
                'minimum_calibration_recall': target_recall,
                'calibration': summarize([(label, score >= toxic_only_threshold) for label, score, _ in calibration]),
                'evaluation': summarize([(label, score >= toxic_only_threshold) for label, score, _ in evaluation]),
            },
        }
    report = {
        'protocol': 'CYBY24 exploratory Toxic-BERT and Vigil cascade',
        'gate_threshold': gate,
        'truncated': cuts,
        'second_stage_calibration_rule': 'Maximum precision at >=95% recall among calibration positives forwarded by the gate',
        'policies': policies,
        'vigil_stage_host_wall_seconds': round(time.perf_counter() - started, 3),
        'limitations': 'Previously explored CYBY24 source. The shared prefix has an unaudited Vigil token budget. Vigil starts a CLI process per call. The host runtime cannot estimate Android latency or battery use.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
