# SMOTE baseline

This is a generic, dataset-agnostic SMOTE experiment for binary fraud classification.

SMOTE is an oversampling technique rather than a classifier, so this implementation reports a complete pipeline:
1. train/test split
2. preprocessing
3. SMOTE applied to the training partition only
4. Logistic Regression classifier
5. evaluation on the untouched test partition

Usage:

```bash
pip install -r requirements.txt
python algorithms/SMOTE/run_smote.py --data /path/to/data.csv --target is_fraud
```

You can change the classifier with the `--classifier` option.

This is a standard SMOTE implementation using imbalanced-learn. It is not claimed to reproduce a specific paper unless the experimental protocol is explicitly matched to that paper.
