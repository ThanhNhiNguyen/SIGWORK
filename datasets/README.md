# SIGWORK Datasets

This folder defines the benchmark datasets for the SIGWORK fraud-detection experiments.

## 1. IEEE-CIS Fraud Detection

Source: Kaggle IEEE-CIS Fraud Detection competition:
https://www.kaggle.com/competitions/ieee-fraud-detection/data

Expected files:
- train_transaction.csv
- train_identity.csv
- test_transaction.csv
- test_identity.csv
- sample_submission.csv

The Kaggle listing currently reports about 1.35 GB for the full dataset and requires accepting the competition rules before download.

Download:

```bash
python datasets/IEEE-CIS/download.py
```

The script uses the Kaggle CLI/API. Configure Kaggle credentials first.

## 2. Elliptic Bitcoin

Source: The Elliptic Bitcoin transaction graph dataset. PyTorch Geometric exposes the three original CSV files through its public data mirror:
https://data.pyg.org/datasets/elliptic

Expected files:
- elliptic_txs_features.csv
- elliptic_txs_classes.csv
- elliptic_txs_edgelist.csv

The standard dataset has 203,769 transactions, 234,355 transaction-flow edges, 49 time steps, and 165 node features. The classification labels are illicit, licit, and unknown.

Download:

```bash
python datasets/Elliptic/download.py
```

## Storage policy

The raw datasets are intentionally NOT committed to GitHub:
- IEEE-CIS is too large and is subject to Kaggle competition rules.
- Elliptic raw data is downloaded reproducibly from the public mirror.

Only download scripts, documentation, and preprocessing code belong in this repository.

## Research protocol

For temporal graph experiments, preserve the original time-step information and use time-aware train/validation/test splits. Do not select thresholds on the final test set.

For heavily imbalanced fraud tasks, report precision, recall, F1, ROC-AUC and PR-AUC in addition to accuracy.
