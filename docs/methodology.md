# Methodology — Historical Experiments

**Task:** Predict six non-exclusive binary labels for each text. Treat this as multi-label classification (not mutually exclusive multi-class classification).

## Split and leakage controls
- Original training file: 159,571 comments.
- Normalize text by lowercasing, trimming and collapsing whitespace to identify near-identical text groups. The historical split kept identical normalized groups together and used multi-label stratification.
- Full training: 111,707; validation: 23,932; internal test: 23,932.
- Historical training subset: 34,044, enriched for positive examples; sampled **only within training**. Model predictions and thresholds use validation, and tests are for reporting.
- The Jigsaw official scored test has 63,978 labeled rows; historical analysis used 63,938 after normalized overlap filtering. Do not evaluate rows where official label values are `-1`.

## Modeling
- Baselines: One-vs-rest Logistic Regression and calibrated Linear SVM with word + character TF-IDF. TF-IDF was fitted using training records only.
- Transformers: DistilBERT and RoBERTa; six independent logits with sigmoid for evaluation. BCE, positive-weighted BCE and focal loss were investigated.
- Checkpoint selection: validation macro Average Precision. Thresholds: per-label F1-maximizing cutoffs on validation. The frozen ensemble averages two model probabilities.

## Metrics
**Primary model-selection:** macro Average Precision (ranking quality, no threshold).
**Primary deployment classification:** macro F1 with per-label thresholds selected on development data.
**Always report:** per-label support, precision, recall, F1, AP; micro F1 and confusion counts; identity-term subgroup false-positive rate with support and uncertainty; compute/inference costs.

## Comparability caveat
Do not rank teammate metrics from different splits, samples, or threshold rules on the same leaderboard. The official test scores have already been seen: future experiments must be selected using development subsets only, and use fresh evaluation data where possible.

## Reference to original work
See `notebooks/` in numbered order and `results/registry.csv`. Notebook outputs were intentionally removed from the public-sharing copy. Evidence tables in `results/` are derived from the historical runs, not generated from the code in this repository as part of the package build.
