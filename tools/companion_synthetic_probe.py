#!/usr/bin/env python3
"""Evaluate compAnIon on the existing synthetic examples using its BERT-CNN architecture."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertModel

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/companion-synthetic.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    digest = hashlib.sha256()
    with args.weights.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    tokenizer = BertTokenizer(vocab_file=str(ROOT / 'app/src/main/assets/model/vocab.txt'), do_lower_case=True)
    encoder = TFBertModel(BertConfig())
    ids = tf.keras.layers.Input(shape=(400,), dtype=tf.int64, name='input_ids')
    types = tf.keras.layers.Input(shape=(400,), dtype=tf.int64, name='token_type_ids')
    mask = tf.keras.layers.Input(shape=(400,), dtype=tf.int64, name='attention_mask')
    sequence = encoder({'input_ids': ids, 'token_type_ids': types, 'attention_mask': mask})[0]
    features = [tf.keras.layers.GlobalMaxPooling1D()(tf.keras.layers.Conv1D(filters, width, activation='relu')(sequence))
                for width, filters in ((3, 100), (4, 100), (5, 50), (10, 25))]
    merged = tf.keras.layers.concatenate(features, axis=1)
    hidden = tf.keras.layers.Dense(200, activation='relu', name='hidden_layer')(merged)
    hidden = tf.keras.layers.Dropout(0.1)(hidden)
    hidden = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer2')(hidden)
    hidden = tf.keras.layers.Dropout(0.1)(hidden)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='classification_layer')(hidden)
    model = tf.keras.Model(inputs=[ids, types, mask], outputs=[output])
    model.load_weights(str(args.weights))
    cases = json.loads((ROOT / 'data/gemma4-prompt-holdout.json').read_text())['cases']
    results = []
    for item in cases:
        batch = tokenizer(item['chat'], max_length=400, truncation=True, padding='max_length', return_tensors='tf')
        score = float(model([batch.input_ids, batch.token_type_ids, batch.attention_mask], training=False).numpy()[0][0])
        results.append({'id': item['id'], 'researcher_label': item['label'], 'score': score, 'flagged_at_0_5': score > 0.5})
    report = {'model': 'chatcompanion/compAnIonv1', 'revision': 'b2f8a04ae9c895acd0f008e67af27ebc2c81641b',
              'sha256': digest.hexdigest(), 'cases': results,
              'limitations': 'Previously examined synthetic examples with author-intended labels. CompAnIon targets grooming rather than all concern categories. No independent framework parity or phone test.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
