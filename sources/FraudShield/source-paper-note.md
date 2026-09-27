Source-paper details from supplied PDF:
- SMOTE is used for class imbalance.
- Feature selection combines Extra Trees, Information Gain Score, and LASSO; 11 top features are selected.
- Voting uses Random Forest, Gradient Boost, Logistic Regression.
- Stacking uses a Decision Tree final estimator.
- CNN uses three Conv1D layers with filter widths 32, 64, 128; Adam lr 0.001.
- LSTM uses two LSTM layers; models are trained for 10 epochs.
- A CNN+LSTM hybrid is also evaluated.
