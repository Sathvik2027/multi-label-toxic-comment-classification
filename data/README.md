# Dataset access — data files are intentionally not committed

1. Download the public Jigsaw Toxic Comment Classification Challenge training and test archives from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge.
2. Historical Colab source references `/content/jigsaw-toxic-comment-classification-challenge.zip` (outer ZIP with nested CSV ZIPs).
3. Processed project splits originally live at `MyDrive/Colab Notebooks/jigsaw-toxic-classifier/data/processed/` as `train_full.csv`, `train_sub.csv`, `val.csv`, `test_internal.csv`, and `test_external.csv`.
4. Keep those row-level CSVs locally/in Drive, **not in GitHub**. Preserve comment IDs and label order when rerunning experiments.
5. Exclude `test_labels.csv` rows whose labels are `-1`. Avoid external test reuse for new model selection.
