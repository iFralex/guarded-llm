# Sarcasm and context: paired synthetic probe

We wrote [18 English examples](../data/sarcasm-context-pairs.json), six each of friendly teasing, hostile sarcasm and direct insults. Each contains three preceding turns and a target message. We evaluated the same target alone and after its preceding turns with Toxic-BERT, RoBERTa and Phocinae. The wording of the target is identical in both views. In one cross-case pair, the exact phrase “Wow, a true fashion icon.” appears in friendly and hostile contexts.

These are researcher-authored examples with author-intended labels. We did not obtain independent annotation. The hostile contexts explicitly describe repeated harassment, so a context-level flag may come from those earlier turns, not the model recognizing sarcasm in the target. When only the target is supplied, hostile intent and repetition may not be inferable. This is a small controlled diagnostic, not a prevalence estimate or an independent evaluation of sarcasm understanding.

The same frozen thresholds from the prior exploratory CYBY24 256-token comparison were used for both views. No thresholds were selected using these examples. Toxic-BERT's threshold was 0.057836, RoBERTa's 0.609913 and Phocinae's 0.025191. Phocinae uses its CYBY24 fine-tuned research checkpoint and the question “Does this conversation contain cyberbullying?” Toxic-BERT is a toxicity model. Their different objectives and score scales affect the comparison. Scores are uncalibrated and the thresholds are not probabilities of risk. The Toxic-BERT app's threshold is 0.5; it produced the same flag counts shown below on this specific set.

| Model | Friendly flags alone / 6 | Friendly flags with context / 6 | Hostile sarcasm flags alone / 6 | Hostile sarcasm flags with context / 6 | Direct insult flags alone / 6 | Direct insult flags with context / 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toxic-BERT | 0 | 0 | 0 | 0 | 6 | 6 |
| RoBERTa | 0 | 1 | 0 | 1 | 6 | 6 |
| Phocinae | 5 | 4 | 2 | 6 | 6 | 6 |

On these examples, context changed four Phocinae hostile cases from unflagged to flagged and one friendly case from flagged to unflagged. Phocinae still flagged four of six friendly cases with context. RoBERTa added one flag in each of the friendly and hostile categories. Toxic-BERT flagged each direct insult and none of the hostile sarcasm cases in either view. Context here includes explicit descriptions of prior conduct, so these observations cannot establish whether a model understood the target's irony or whether adding context improves real-world performance.

The [case-level report](sarcasm-context-results.json) records the inputs, scores, decisions, frozen thresholds, and source dataset SHA-256. The runner asserts that each input fits the token budgets of the three models, so none of these inputs was truncated. The host Mac used CPU inference; the context mode is an experiment and has not been verified as the app's deployed acquisition or context pipeline.

To reproduce, prepare the pinned models and dependencies as described in [research.md](research.md), including the CYBY24 Phocinae trained subset and original encoder checkpoint. From the project root run:

```sh
PYTHONPATH="$PWD/.modern-deps:/path/to/benchmark-deps" \
PHOC_RUNTIME="$PWD/.cache/phocinae-server" HF_HOME="$PWD/.cache/huggingface" \
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0 \
python tools/sarcasm_context_probe.py
```

The runner refuses to overwrite an existing report. The full dataset is synthetic and the stored conversations are intended for public review.
