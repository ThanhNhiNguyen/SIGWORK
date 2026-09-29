# IEEE-CIS Fraud Detection

Canonical source:
https://www.kaggle.com/competitions/ieee-fraud-detection/data

The competition data contains separate transaction and identity tables joined by TransactionID. The target in the training transaction table is `isFraud`.

Because the dataset is hosted behind Kaggle's competition rules, this repository stores only reproducible download instructions and does not redistribute the raw files.

## Setup

Install the Kaggle CLI:

```bash
pip install kaggle
```

Authenticate with a Kaggle API token, then run:

```bash
python datasets/IEEE-CIS/download.py
```
