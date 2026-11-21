# NLP-04 · Multi-Label Toxic Comment Classification with Explainability

Research project on the Jigsaw Toxic Comment Classification Challenge (six independent labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

> **Status:** Completed exploratory baselines, transformer comparisons, frozen ensemble evaluation, initial explainability and fairness audits. Additional controlled experiments and the deployment brief remain in progress. Do not interpret the current official Jigsaw test as an untouched selection set for newly designed models.

## Project objectives

1. Compare classical lexical baselines with six-label transformer classifiers.
2. Investigate class imbalance (BCE, moderated weighted BCE and focal loss).
3. Optimize class-specific decision thresholds on **development data**, and report macro AP, macro F1, per-label precision/recall/F1/AP.
4. Analyze false positives, especially benign identity-associated comments, and explain selected predictions using LIME.
5. Evaluate whether ensembling meaningfully adds value over a simpler standalone model.

## Dataset and split (historical experiments)

- Data: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
- Jigsaw training file: **159,571 rows**.
- Duplicate-group-aware partition: **111,707 full-training**, **23,932 validation**, **23,932 internal-test**.
- Previous model training used **34,044 training rows** (all 11,348 positive-any-label training comments + 22,696 sampled clean comments). This is an **enriched training subset**, not a representative prevalence sample.
- The external scored test has **63,978 labeled rows**, with **63,938** rows retained after overlap filtering for this project's historical evaluation.
- Original Kaggle `test_labels.csv` contains `-1` rows that must not be used as ground truth.
- **Neither raw Kaggle files nor row-level processed comment texts are included in this repository.**

## Historical results (same development protocol)

| Run | Model | Validation macro AP |
|---|---|---:|
| A1 | Logistic Regression + word/character TF-IDF | 0.6831 |
| A2 | Calibrated Linear SVM + word/character TF-IDF | 0.6715 |
| B1_128 | DistilBERT BCE (128 tokens) | 0.7283 |
| B1_256 | DistilBERT BCE (256 tokens, checkpoint recovery caveat) | 0.7307 |
| C1_128 | DistilBERT weighted BCE (128 tokens) | 0.7314 |
| C2_128 | DistilBERT focal loss (128 tokens) | 0.7333 |
| R1_128 | RoBERTa focal loss (128 tokens) | 0.7403 |
| Final frozen ensemble | RoBERTa focal + DistilBERT BCE-256 | **0.7551** |

Final *historical* ensemble: validation tuned macro F1 **0.7100**; internal test macro F1 **0.6814**; filtered external Jigsaw test macro F1 **0.6097**. The external F1 gain over standalone RoBERTa is small and uncertain; do not claim proven superiority.

## Repository navigation

- [`notebooks/`](notebooks/): source code of completed experiments, numbered in research order. Notebook **outputs and execution counts have been removed** from this public-sharing copy (to avoid accidentally exposing comment text or notebook secrets). Original executed notebooks are retained separately.
- [`src/`](src/): Python helper modules used by Colab notebooks.
- [`results/`](results/): aggregated, reviewed CSV/JSON tables only; no raw text or model weights.
- [`docs/methodology.md`](docs/methodology.md): methodology and split rules.
- [`docs/status_and_limitations.md`](docs/status_and_limitations.md): completed work vs. pending work.
- [`data/README.md`](data/README.md): how to obtain the underlying Jigsaw archive.
- [`scripts/check_public_repo.py`](scripts/check_public_repo.py): a lightweight pre-push safety audit.

## Run in Google Colab

These notebooks originated in a Google Drive working directory (`MyDrive/Colab Notebooks/jigsaw-toxic-classifier`). The **data files and private checkpoints live in Drive**, not this public repository. Before running, mount Drive, obtain the Kaggle archive and recreate or restore the processed data and checkpoints. Original cells may contain project-specific Drive paths; review them rather than assuming a local clone runs unchanged. The new `18_value_add_phase2.ipynb` reads the existing processed split and saved prediction files and does not retrain transformers.

Install packages using `pip install -r requirements.txt`, subject to the versions compatible with your chosen Colab runtime. Export `pip freeze` and CUDA/GPU details with final experimental results; the requirements here are not a pinned lockfile.

## Interpretation / limitations

- The existing transformer loss-function improvements are small; do not claim definitive statistical superiority without controlled reruns.
- The 256-token DistilBERT run had a checkpoint-recovery caveat; it is not a clean length ablation.
- Macro Average Precision and tuned macro F1 describe different aspects of the model.
- Identity-term false positives are concerning but keyword associations alone are not causal bias proofs.
- Final official test scores were already inspected. **New model selection must use development data**, with fresh untouched evaluation if available.
- All work is research/prototyping; predictions should support human moderation, not automatically punish users.

## Current next steps

See [`docs/status_and_limitations.md`](docs/status_and_limitations.md) and the planned Phase-2 notebook. This repo does **not** claim Phase-2 training or fairness mitigation was completed unless its results are subsequently added with evidence.

## Citation / related work

- Davidson et al. (2017), *Automated Hate Speech Detection and the Problem of Offensive Language*.
- van Aken et al. (2018), *Challenges for Toxic Comment Classification: An In-Depth Error Analysis*.
- Jigsaw / Conversation AI dataset: [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

## Privacy and ownership

Do not commit the original Jigsaw ZIP/CSV rows, rendered comments from LIME, API keys, tokens, private paths, model weights, or teammates' files without approval/attribution. Review the notebook source before publishing. Contributions and authorship should reflect actual work; do not present a teammate's model as your own.
