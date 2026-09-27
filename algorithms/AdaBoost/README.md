# AdaBoost baseline

Generic AdaBoost classifier for binary financial-fraud classification.

Usage:

```bash
pip install -r requirements.txt
python algorithms/AdaBoost/run_adaboost.py --data /path/to/data.csv --target is_fraud
```

The implementation uses scikit-learn's AdaBoostClassifier and evaluates the untouched test set using Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC.

This is a standard AdaBoost implementation and is not claimed to reproduce a particular paper unless the paper's exact preprocessing, learner, hyperparameters and evaluation protocol are matched.
