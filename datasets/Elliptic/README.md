# Elliptic Bitcoin Fraud / Illicit Transaction Dataset

Canonical dataset files:
- elliptic_txs_features.csv
- elliptic_txs_classes.csv
- elliptic_txs_edgelist.csv

Public PyTorch Geometric mirror:
https://data.pyg.org/datasets/elliptic

Reference structure:
- features: transaction id, time step, 165 numerical features
- classes: transaction id and class (unknown, licit, illicit)
- edgelist: directed transaction-flow edges

The standard release contains 203,769 nodes, 234,355 edges, 49 time steps and 165 features.

Use this dataset for graph and temporal fraud experiments where the method's input assumptions are compatible.
