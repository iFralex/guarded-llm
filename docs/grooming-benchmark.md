# Grooming and early detection evidence

The experiments below use PAN12. A positive example means a conversation prefix associated with an author flagged by the corpus. For a deployment-style benchmark, use annotations that mark when a conversation first contains enough evidence to justify an alert.

| Candidate | Data and setup | TP | FP | FN |
| --- | --- | ---: | ---: | ---: |
| Laya | Exploratory 10-turn development probe, 209 examples | 31 | 171 | 4 |
| Verdict | Exploratory 10-turn development probe, 209 examples | 25 | 171 | 10 |
| ModernBERT | Frozen encoder and trained head, 10-turn development probe, 209 examples | 34 | 19 | 1 |
| MobileBERT | Fine-tuned, matched development, 563 examples | 60 | 37 | 10 |
| Phocinae | Fine-tuned, matched development, 563 examples | 64 | 22 | 6 |
| Phocinae | Frozen selection, filtered official test, 2,943 examples | 55 | 138 | 7 |

Compare models directly only within matched runs. The two development settings differ in samples, context budget and training. The Phocinae official test was previously evaluated and is now a historical reference. The previous audit excluded training-source authors and exact text overlap and ran a partial near-duplicate filter. Model pretraining sources and 174 short examples that escaped that filter remain uncertain. See the [research notes](research.md) and [test report](pan12-matched-test.json).

| Proposed candidate | Evidence and overlap to check |
| --- | --- |
| compAnIon v1 | The authors report PAN12 training and PAN12 test use in their model data inventory. Treat both partitions as exposed at project level until the exact checkpoint's training, calibration and evaluation manifests are confirmed. Check overlap with the authors' other positive source, PJZC. [Author repository](https://github.com/chatcompAnIon/chatcompAnIon) |
| Vogt et al. early detection models | The authors publish trained models and code. PANC draws on PAN12 and ChatCoder2. Verify the provenance and split of each checkpoint before comparing it on either dataset. [Paper](https://aclanthology.org/2021.acl-long.386/) · [Project artifacts](https://early-sexual-predator-detection.gitlab.io/) |
| Llama Guard 3-1B | Candidate for a generic safety baseline. Freeze how its categories map to the task before opening labels, and use abstention where the output cannot support the requested distinction. [Model card](https://huggingface.co/meta-llama/Llama-Guard-3-1B) |

Before a new independent run, have a teammate retain the test labels. Record each candidate's training and evaluation sources, group by conversation and participant where possible, and audit normalized exact matches and near duplicates across released source partitions. Mark each model–dataset pairing as independent, previously exposed, or unverified based on that evidence. Report conversation-level recall, false alarms, and delay from the first annotated actionable turn. Use the same chronological prefixes and fixed decision policies for all candidates.

## Matched exploratory grooming and bullying pilot

We ran compAnIon, the authors' quantized PANC BERT and MobileBERT, fine-tuned Phocinae and quantized Llama Guard on shared 128-token-bounded prefixes. For each task the same 40 calibration and 40 evaluation examples were drawn using fixed seeds, balanced 20 positive and 20 negative in each partition. PAN12 uses the previous author-grouped calibration and development subsets, with author-derived weak labels. CYBY24 uses the previously explored grouped calibration and test partitions. All PAN12 inputs have at most ten messages; CYBY24 uses the beginning of each thread. The common prefix was truncated on 9 of 80 PAN12 examples and 64 of 80 CYBY24 examples.

For scored models, a threshold maximizes calibration precision subject to at least 0.8 recall. Llama Guard uses a fixed generic safety prompt and explicit safe/unsafe output. Invalid UTF-8 and unparseable responses are abstentions; one calibration negative on the bullying task abstained. Phocinae uses the existing task-specific fine-tuned checkpoint for each dataset. Other checkpoints were used as released. The PANC checkpoints were selected from the authors' Android demo; the calibrated threshold can differ substantially from their native decision. Results on each 40-example evaluation sample follow.

| Task | Model | TP | FP | TN | FN |
| --- | --- | ---: | ---: | ---: | ---: |
| Grooming proxy | compAnIon | 19 | 1 | 19 | 1 |
| Grooming proxy | eSPD BERT PANC | 14 | 1 | 19 | 6 |
| Grooming proxy | eSPD MobileBERT PANC | 15 | 0 | 20 | 5 |
| Grooming proxy | Phocinae | 15 | 0 | 20 | 5 |
| Grooming proxy | Llama Guard | 0 | 6 | 14 | 20 |
| Bullying | compAnIon | 14 | 16 | 4 | 6 |
| Bullying | eSPD BERT PANC | 17 | 18 | 2 | 3 |
| Bullying | eSPD MobileBERT PANC | 20 | 18 | 2 | 0 |
| Bullying | Phocinae | 17 | 8 | 12 | 3 |
| Bullying | Llama Guard | 6 | 3 | 17 | 14 |

These are small, deliberately balanced and already-explored samples. The PAN12 positives identify conversations associated with a flagged author; they do not establish that a particular prefix shows grooming. compAnIon and PANC models have PAN12-related training sources, and Phocinae was trained and selected using related partitions. These results are not independent estimates and cross-model overlap differs. CYBY24 is public Twitter, with 64 of 80 inputs truncated. Generic safety labels do not directly match either task, and no phone performance was measured. All Llama Guard outputs on evaluation parsed, with one invalid calibration output on bullying. See the [aggregate report](matched-grooming-bullying-pilot.json).

The lab source at commit `76eff667a0d1f27e5091016483cb69522a7319a0` constructs BERT features via TensorFlow Model Maker's tokenizer. The Android demo uses BertNLClassifier at sequence length 512. The Google BERT reference tokenizer at commit `eedf5716ce1268e56f0a50264a88cafad334ac61` produced identical padded token IDs to our Hugging Face tokenizer for all 22 synthetic inputs and the 160 untruncated pilot texts, with the demo vocabulary. This checks token IDs on those inputs; it does not establish end-to-end Android or Model Maker parity, or explain the synthetic model decisions.

The verified eSPD archive has 18 named run configurations and 24 TensorFlow Lite files, including BERT and MobileBERT quantized and non-quantized variants for PANC and VTPAN, plus BERT-large runs. The selected demo revisions are `2020-11-21_23-11-03__bert_classifier_on_PANC_with_seq-len-512` and `2020-11-30_17-08-16__mobilebert_classifier_on_PANC_with_seq-len-512`, at Android demo commit `114de976d2f6b2385d90d20fea5def2f60e101ab`. Their model SHA-256 digests are `fe734331c4087e1fb81b8b7a9ab80ff29337db5df57b676fd5460a605c810447` and `78a8b2565ddf9837d0229cf67dc530b8bf46072903168d41919fef324ac7c629`. The first BERT pilot erroneously selected the MobileBERT folder; both BERT runs were discarded and repeated after correcting the selection.

Reproduce with `tools/compare_grooming_bullying.py --dataset grooming|bullying --model companion|espd_bert|espd_mobile|phocinae|llamaguard`, local datasets, the pinned checkpoints and dependencies. Source messages are processed locally and reports contain aggregate counts only. Llama Guard's executable needs access to its local inference port in this environment.
