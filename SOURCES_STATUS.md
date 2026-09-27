# SIGWORK Source Status

Updated 2026-09-27.

## Public repositories pinned as Git submodules

| Method | Repository | Status |
|---|---|---|
| FraudGNN-RL | https://github.com/Siddartha22/FraudGNN-RL | Public code candidate; not verified as authors' official paper implementation |
| QGNN | https://github.com/estasz/GNN-QGNN-FraudDetection | Public GNN/QGNN fraud project; not verified as the exact paper implementation used in SIGWORK |
| Fraud Shield | https://github.com/rohith-chitturi/FraudShield_AI | General fraud platform; not verified as the supplied paper's implementation |
| Fraud Shield | https://github.com/Omensah-15/FraudShield | SMOTE + RandomForest fraud project; not verified as the supplied paper's implementation |
| Hydro-TGT | https://github.com/mannetej11-gif/Hydro-TGT-Blockchain-Fraud-Detection | Public project presenting a GraphSAGE + temporal Transformer architecture; README describes a development roadmap/scaffold |

## Not pinned

- GraphFA: no verified official repository found in the current GitHub search. The uploaded GraphFA paper is the implementation specification.
- HIF-FD: the repository named TILab-Hi/HiFD is a face de-identification project, not financial fraud, and must not be used for HIF-FD.
- MRV-RSA: no repository found in the current GitHub search.
- FinDEx: no repository found in the current GitHub search.

## QGNN repository notes

The pinned QGNN project README describes:
- Classical GNN using GraphSAGE/GCN layers.
- QGNN using SGConv followed by a simulated quantum layer with RX/RY gates.
- CUDA-Q simulation.
- Fraud metrics including precision, recall, F1, ROC-AUC and PR-AUC.

## Fraud Shield paper specification

The supplied paper describes:
- PCA preprocessing.
- SMOTE balancing.
- Normalization.
- Feature selection using Extra Trees, Information Gain Score and LASSO, with 11 selected features.
- Voting classifier: Random Forest + Gradient Boost + Logistic Regression.
- Stacking classifier: Decision Tree final estimator.
- CNN, LSTM, and CNN+LSTM hybrid models.

## Getting the source code locally

From a clone of SIGWORK:

```bash
git clone https://github.com/ThanhNhiNguyen/SIGWORK.git
cd SIGWORK
git submodule update --init --recursive
```

This pulls the pinned public repositories into `sources/`.

Do not report a repository as the original author implementation unless the repository's authorship and architecture have been verified against the corresponding paper.
