# Synthetic pilot and conversation length

The new [synthetic dataset](../data/synthetic-new-pilot.json) contains 48 English conversations, authored for this experiment. Grooming and bullying each have eight calibration and sixteen evaluation examples, balanced within each split. Its SHA-256 is `92cb94a5fab4512b0a125300c0bfafc20193ccbaadede8ce277f52c71f009fb9`. The same researcher designed the examples, labels and evaluation protocol. The test source was created separately from earlier corpora and is not a blinded or independently annotated benchmark. A second annotator should review the labels and task definitions before using these examples for model selection. No private conversations were used to create the dataset.

The [overlap audit](synthetic-overlap-audit.json) found no normalized exact conversation matches against the locally reconstructed PAN12 training rows, filtered CYBY24 threads and two earlier synthetic sets. The highest five-word-shingle Jaccard similarity was 0.12, and no pair within the new set reached 0.5. This check does not rule out paraphrases, overlap with other datasets or unknown model pretraining data. The author deliberately included repeated themes, so lexical disjointness does not imply independent concepts.

The same source texts were used for seven candidates. All synthetic texts fit a common prefix bounded by 384 tokens, including the Phocinae question, with no truncation. For scored models, the threshold was selected using the eight calibration examples to maximize precision at at least 0.8 recall. With four positive calibration cases the threshold is highly unstable. Llama Guard uses its fixed generic safety prompt and safe/unsafe decision. None of these scores is a calibrated probability.

| Model | Grooming TP / 8 | Grooming FP / 8 | Bullying TP / 8 | Bullying FP / 8 |
| --- | ---: | ---: | ---: | ---: |
| compAnIon | 8 | 8 | 6 | 5 |
| eSPD BERT PANC | 6 | 7 | 8 | 4 |
| eSPD MobileBERT PANC | 5 | 7 | 8 | 6 |
| Phocinae | 7 | 5 | 5 | 0 |
| Toxic-BERT | 4 | 2 | 8 | 6 |
| RoBERTa | 3 | 3 | 5 | 4 |
| Llama Guard | 4 | 1 | 0 | 0 |

Phocinae uses task-specific previously fine-tuned checkpoints. The generic Llama Guard taxonomy and prompt do not exactly match the two labels. These tiny author-labeled samples cannot establish real-world sensitivity, false-alarm rate, fairness, early-detection ability or ranking. See [synthetic aggregate results](synthetic-pilot-results.json), which include calibration counts and thresholds.

## Longer context on CYBY24

We reused the prior exploratory balanced 40 calibration and 40 evaluation threads, and compared common 128- and 384-token prefixes for the same candidates. This source has already been used for model development. For each window and each score-based model, thresholds were reselected from the same calibration partition by the same rule. Consequently the comparison combines changing the context window with recalibration. Llama Guard retains the same fixed prompt and decision rule. The 128-token setup truncated 64 of 80 threads; the 384-token setup truncated 10 of 80. These totals combine calibration and evaluation.

| Model | TP / 20 at 128 | FP / 20 at 128 | TP / 20 at 384 | FP / 20 at 384 |
| --- | ---: | ---: | ---: | ---: |
| compAnIon | 14 | 16 | 13 | 16 |
| eSPD BERT PANC | 17 | 18 | 17 | 18 |
| eSPD MobileBERT PANC | 20 | 18 | 19 | 15 |
| Phocinae | 17 | 8 | 17 | 8 |
| Toxic-BERT | 13 | 9 | 15 | 8 |
| RoBERTa | 14 | 14 | 15 | 11 |
| Llama Guard | 6 | 3 | 11 | 10 |

There was one invalid Llama Guard calibration output for each window and no evaluation abstention. Its tokenizer and prompt budget were not independently audited. Results are from a small balanced sample of public Twitter threads and do not estimate Instagram prevalence or support a deployment decision. Model training objectives, histories and architectures differ. See [context aggregate results](cyby24-context-comparison.json).

## Reproduction

Obtain the pinned model artifacts and local runtime documented in [grooming-benchmark.md](grooming-benchmark.md), plus the generated Toxic-BERT ONNX assets and the local CYBY23, CYBY24 and PAN12 inputs. The pipeline uses Phocinae, BERT and RoBERTa tokenizers to ensure their common prefix fits each tested budget. It caps the budget at 384 to fit compAnIon's 400-token input. It operates locally and aggregate reports for real datasets contain no messages or identifiers. The fixed Llama Guard prompt is defined in `tools/llama_guard_synthetic_probe.py`.

```sh
export PHOC_RUNTIME="$PWD/.cache/phocinae-server"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false MPLCONFIGDIR=/tmp/guarded-matplotlib
export PYTHONPATH="$PWD/tools:/path/to/tensorflow-runtime:$PWD/.modern-deps:/tmp/guarded-llm-benchmark-deps"
python tools/audit_synthetic_overlap.py
python tools/compare_grooming_bullying.py --dataset synthetic_grooming --model companion --budget 384 --run-tag synthetic
python tools/compare_grooming_bullying.py --dataset bullying --model companion --budget 384 --run-tag longcontext
```

The evaluator refuses to overwrite existing result files. Use `synthetic_bullying` for the second synthetic task, replace the model argument with each candidate, and use budgets 128 and 384 for the CYBY24 length comparison. The overlap audit uses the default PAN12 training-only pilot partition reconstruction. Llama Guard requires a working local llama.cpp executable.
