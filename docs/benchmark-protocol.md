# English chat safety benchmark

The first run is an exploratory engineering check on CYBY24. Its 549 calibration and 551 evaluation threads were reconstructed with the existing grouped split. The corpus has already been used during development. Run Toxic-BERT, RoBERTa, ModernBERT and Phocinae first. Toxic-BERT scores the common thread as a toxicity baseline. RoBERTa uses the published fine-tuned checkpoint. ModernBERT and Phocinae use the existing CYBY24 checkpoints. Record the training regime beside each result.

Use the same thread prefix for all four models, within 256 tokens for each tokenizer including Phocinae's question. Record how often this removes text. Use single-thread inference in both partitions. For models with scores, select the threshold with maximum calibration precision at or above 80% recall. Break ties by higher recall, then a higher threshold. If no setting meets the recall floor, report that outcome. For fixed decisions, freeze the prompt or category mapping before evaluation. Report counts, precision, recall, false-positive rate, and coverage. Abstentions count against recall on positives. Estimate false alerts per 1,000 threads at 0.1%, 1% and 5% hypothetical prevalence using the measured false-positive rate and the assumed number of negatives. These scenarios hold source-specific error rates fixed and do not estimate actual Instagram prevalence.

Run each candidate with local assets prepared as in [research.md](research.md).

```sh
export PHOC_RUNTIME="$PWD/.cache/phocinae-server"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0
for model in toxic roberta modernbert phocinae; do
    python tools/cyby24_matched_extended.py --standardized --model "$model" || exit
done
```

Aggregate-only reports go to `reports/cyby24-standardized-*.json`. The reference run used Python 3.12.9, PyTorch 2.7.0, Transformers 4.48.3, Tokenizers 0.21.4, ONNX Runtime 1.20.1, scikit-learn 1.5.2, NumPy 1.26.4, openpyxl 3.1.5, safetensors 0.5.3 and Hugging Face Hub 0.30.2. Keep the same dependency versions and model revisions when comparing runs.

For the independent benchmark, agree on sources and annotation with the professor. Separate bullying and grooming by label and evaluation unit. Keep ConvAbuse as a separate abuse transfer check. Use conversation and participant identifiers where available and audit near-duplicates before splitting. Ask a teammate to retain the independent test labels until the model list, prompts, category mapping, thresholds and comparison rules are frozen. Report sample size, known overlap with training data, and uncertainty around error rates. Include explicit coverage for disagreement and ambiguity. For early grooming detection, evaluate only the turns available at each prediction point and report how early a correct alert appears.

The separate [grooming evidence table](grooming-benchmark.md) records which candidates have been run, what their labels measure, and which model–dataset overlaps still need verification.

Once the standalone run is verified, compare Toxic-BERT or RoBERTa alone with the same first stage followed by Gemma or Vigil. Choose which cases to forward on calibration, including an option to forward borderline cases. Compare false positives at comparable recall and report positives lost at the gate, second-stage call rate and abstentions. Evaluate Llama Guard 3-1B as a generic safety baseline with a prespecified mapping, and audit availability and training overlap of compAnIon and early detection candidates before adding them to the grooming evaluation.

## First exploratory run

All four models completed calibration and evaluation. The common prefix truncated 148 calibration threads and 174 evaluation threads. There were 112 positives and 439 negatives in evaluation. Every model returned a decision for every thread. The last column applies the observed CYBY24 false-positive rate to 990 negatives per 1,000 threads at an assumed prevalence of 1%.

| Model | Detected positives | Missed positives | False positives | Estimated false alerts per 1,000 |
| --- | ---: | ---: | ---: | ---: |
| Toxic-BERT | 82 | 30 | 223 | 503 |
| RoBERTa | 75 | 37 | 255 | 575 |
| ModernBERT | 88 | 24 | 236 | 532 |
| Phocinae | 96 | 16 | 260 | 586 |

At these thresholds Toxic-BERT flagged fewer negatives but also detected fewer positives than ModernBERT and Phocinae. Phocinae had the highest recall. The estimates depend on the CYBY24 error rates and the hypothetical prevalence. See the [aggregate report](cyby24-standardized.json) for all operating points and scenarios.

## Toxic-BERT and Vigil cascade

Keep the Toxic-BERT threshold from the standalone benchmark. Only threads that reach that threshold go to Vigil, which receives the same conversation prefix. An alert requires both stages to flag the thread. Evaluate two Vigil policies. The first uses the publisher's bullying threshold. The second selects maximum precision while retaining at least 95% of positives forwarded on calibration. The 95% requirement applies to the positives seen by Vigil. For each policy, compare a separate Toxic-BERT-only threshold selected on calibration using the cascade's achieved recall as a minimum, so that a simple threshold change gets a fair comparison. Actual recall can differ on evaluation.

With the same offline runtime and assets as above, run:

```sh
python tools/cyby24_cascade.py \
    --vigil-model /path/to/vigil-f16.gguf \
    --vigil-cli .cache/vigil-llama/build-arm64/bin/llama-laya-cli
```

The runner verifies that the Toxic-BERT confusion matrices reproduce the reference and that the Vigil checkpoint matches its pinned hash. The output contains aggregate results. It separates positives missed at the first stage from positives vetoed by Vigil, and reports false alarms removed and the stage-two call rate. Host time includes a fresh Vigil CLI process for each forwarded thread.

| Policy on the 551 evaluation threads | Detected positives | Missed positives | False positives |
| --- | ---: | ---: | ---: |
| Original Toxic-BERT | 82 | 30 | 223 |
| Cascade with Vigil's publisher threshold | 62 | 50 | 178 |
| Toxic-BERT alone, calibrated to the publisher cascade's recall floor | 73 | 39 | 183 |
| Cascade with Vigil threshold calibrated for 95% retention | 71 | 41 | 203 |

The Toxic-BERT-only comparison for the calibrated cascade selected the original Toxic-BERT threshold. Vigil reviewed 305 of 551 evaluation threads (55.4%). The first stage missed 30 positives. The publisher threshold removed 45 false alarms and rejected 20 forwarded positives. The calibrated threshold removed 20 false alarms and rejected 11 forwarded positives. It reached the retention goal on calibration and retained 71 of 82 forwarded evaluation positives (86.6%).

The publisher cascade avoided five more false positives than the stricter Toxic-BERT baseline and detected 11 fewer positives. The calibrated cascade also traded away detection to reduce alerts. These trade-offs and the call rate do not support adopting this cascade at the tested thresholds. The experiment used the previously explored CYBY24 split and the Mac CPU. See the [aggregate results](cyby24-cascade-vigil.json).

## Full-split standalone extensions

MobileBERT, Gemma and Vigil use the same two partitions and raw prefixes as the first run. MobileBERT uses its saved CYBY24 checkpoint, and Vigil uses the publisher's bullying score as the scalar to calibrate. Both use the same recall-floor rule as the other scored models. Gemma uses its frozen three-way bullying prompt with greedy decoding and no thinking. UNCERTAIN and invalid output count as abstentions, with a separate tally for each. Scores from the classifiers are decision scores. Recall counts all positive examples in its denominator, including abstentions. Prevalence projections use observed rates for explicit alerts.

Run the same command above with `--model mobilebert`, `--model vigil` or `--model gemma`. Gemma uses LiteRT-LM API 0.19.1. Vigil uses the pinned CLI and checkpoint documented in the research notes. Their native token budgets were not audited against the other tokenizers, and the saved prefix is shared.

| Model | TP | FP | TN | FN | Uncertain positive | Uncertain negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toxic-BERT | 82 | 223 | 216 | 30 | 0 | 0 |
| RoBERTa | 75 | 255 | 184 | 37 | 0 | 0 |
| ModernBERT | 88 | 236 | 203 | 24 | 0 | 0 |
| Phocinae | 96 | 260 | 179 | 16 | 0 | 0 |
| MobileBERT | 97 | 349 | 90 | 15 | 0 | 0 |
| Vigil | 80 | 278 | 161 | 32 | 0 | 0 |
| Gemma | 91 | 305 | 97 | 11 | 10 | 37 |

Gemma also produced 44 uncertain responses on calibration, for 91 across both partitions. No output was invalid. Its fixed prompt and three-way decision differ from the classifiers' calibrated scores. None of the models produced a low false-alarm rate in this exploratory source. The underlying CYBY24 test split was already inspected and is not an independent holdout. The [aggregate report](cyby24-standardized.json) includes calibration, evaluation and prevalence scenarios.
