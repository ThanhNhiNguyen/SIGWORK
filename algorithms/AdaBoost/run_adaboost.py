"""Run a generic AdaBoost fraud-detection baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    transformers = []
    if numeric:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric)
        )
    if categorical:
        transformers.append(
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical)
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def make_adaboost(random_state: int) -> AdaBoostClassifier:
    weak_learner = DecisionTreeClassifier(
        max_depth=1,
        random_state=random_state,
    )
    # estimator= is the current scikit-learn API; algorithm is intentionally
    # omitted for compatibility with newer releases.
    return AdaBoostClassifier(
        estimator=weak_learner,
        n_estimators=200,
        learning_rate=0.5,
        random_state=random_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="results/adaboost_results.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target!r} not found.")

    y = df[args.target]
    X = df.drop(columns=[args.target])

    if y.nunique(dropna=True) != 2:
        raise ValueError("This implementation expects exactly two target classes.")

    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("classifier", make_adaboost(args.random_state)),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        ),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
