# Project Status and Next Tasks

## Confirmed historical work (KEEP)
- Data audit and duplicate-group-aware split.
- TF-IDF Logistic Regression and calibrated SVM baselines.
- DistilBERT (128/256 BCE, 128 weighted BCE, 128 focal) and RoBERTa (128 focal).
- Validation-selected two-transformer ensemble with six saved thresholds.
- Internal/external evaluation with confidence intervals.
- Twenty LIME example explanations and an initial faithfulness test.
- Identity-term real-comment and counterfactual false-positive audit.

## Caveats / reporting corrections
- Enriched 34K training subset is a speed-vs-representativeness trade-off, not a proven best training strategy.
- DistilBERT 256 involved optimizer/scheduler reset on checkpoint recovery.
- Focal alpha=0.25 does **not** automatically emphasize positive rare examples.
- Loss differences are small; selected configurations require careful uncertainty statements.
- The ensemble's external macro F1 gain over RoBERTa is very small.
- Existing official test labels were already examined; avoid selecting future configurations based on test performance.
- Fairness subgroup associations need sample support, context review, and limitation statements.

## To finish (not yet verified as complete)
1. Confirm that the frozen model checkpoints can be reloaded (not just `*.npy` predictions).
2. Review and manually tag a representative subset of error cases (original `errors_to_tag.csv` stays private).
3. Draft moderation deployment/monitoring brief; do not claim an app is deployed.
4. Freeze reproducible environment information, record output provenance, and check GitHub release for private data.
5. Add Phase-2 controlled sampling/fusion experiments **only after running them**, along with new metrics and notebooks.

## Strong optional improvement experiments
1. Enriched 34K vs representative 34K vs full 111K under identical word TF-IDF (cheap sampling ablation).
2. Full-data DistilBERT BCE vs existing enriched-data DistilBERT BCE-128 (GPU, if compute available).
3. Calibrated lexical–transformer fusion on independent development splits (only if measurable gain).
4. Fairness mitigation if an appropriate controlled setup and reliable measurements are available.

Do not treat unrun experiments as results.
