"""Compare traditional ML classifiers for Eclipse bug-priority prediction."""

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from workflow_utils import (
    CATEGORICAL_COLUMNS,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    FEATURE_COLUMNS,
    TEXT_COLUMNS,
    create_output_dir,
    load_dataset,
    save_dataset_profile,
    save_run_summary,
    save_split,
    split_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate traditional models for bug-priority prediction."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    return args


def make_pipeline(classifier: Any) -> Pipeline:
    features = ColumnTransformer(
        [
            (
                f"{column.lower()}_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
                column,
            )
            for column in TEXT_COLUMNS
        ]
        + [
            (
                "metadata",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            )
        ],
        remainder="drop",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def make_models(labels: list[str], seed: int) -> dict[str, Pipeline]:
    return {
        "majority_baseline": make_pipeline(
            DummyClassifier(strategy="most_frequent")
        ),
        "decision_tree": make_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=seed)
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
        ),
        "linear_svm": make_pipeline(
            LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)
        ),
        "naive_bayes": make_pipeline(
            MultinomialNB(class_prior=np.full(len(labels), 1 / len(labels)))
        ),
    }


def score(
    model_name: str,
    labels: list[str],
    actual: pd.Series,
    predicted: np.ndarray,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "Model": model_name,
        "Accuracy": accuracy_score(actual, predicted),
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"],
    }
    per_class = [
        {
            "Model": model_name,
            "Split": split,
            "Priority": label,
            "Precision": report[label]["precision"],
            "Recall": report[label]["recall"],
            "F1": report[label]["f1-score"],
            "Support": report[label]["support"],
        }
        for label in labels
    ]
    return summary, per_class


def main() -> None:
    args = parse_args()
    data, labels = load_dataset(args.input)
    train_indices, test_indices = split_dataset(data, args.test_size, args.seed)
    train_x = data.loc[train_indices, FEATURE_COLUMNS]
    test_x = data.loc[test_indices, FEATURE_COLUMNS]
    train_y = data.loc[train_indices, "Priority"]
    test_y = data.loc[test_indices, "Priority"]
    if train_y.value_counts().min() < args.folds:
        raise ValueError("Each class needs at least --folds training examples.")

    output_dir = create_output_dir(args.output_dir)
    save_dataset_profile(data, output_dir)
    save_split(data, train_indices, test_indices, output_dir)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cv_metrics: list[dict[str, Any]] = []
    cv_per_class: list[dict[str, Any]] = []
    test_metrics: list[dict[str, Any]] = []
    test_per_class: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {
        "Bug_ID": data.loc[test_indices, "Bug_ID"].to_numpy(),
        "Actual Priority": test_y.to_numpy(),
    }
    fitted_models: dict[str, Pipeline] = {}

    for name, model in make_models(labels, args.seed).items():
        print(f"Evaluating {name}...")
        cv_pred = cross_val_predict(
            model, train_x, train_y, cv=cv, method="predict", n_jobs=1
        )
        summary, classes = score(name, labels, train_y, cv_pred, "cross-validation")
        cv_metrics.append(summary)
        cv_per_class.extend(classes)

        model.fit(train_x, train_y)
        test_pred = model.predict(test_x)
        summary, classes = score(name, labels, test_y, test_pred, "held-out test")
        test_metrics.append(summary)
        test_per_class.extend(classes)
        predictions[f"{name} Prediction"] = test_pred
        pd.DataFrame(
            confusion_matrix(test_y, test_pred, labels=labels),
            index=labels,
            columns=labels,
        ).rename_axis("Actual\\Predicted").to_csv(
            output_dir / f"confusion_matrix_{name}.csv"
        )
        fitted_models[name] = model

    cv_frame = pd.DataFrame(cv_metrics).sort_values(
        ["Macro F1", "Weighted F1"], ascending=False
    )
    cv_frame.to_csv(output_dir / "cross_validation_metrics.csv", index=False)
    pd.DataFrame(cv_per_class).to_csv(
        output_dir / "cross_validation_per_class.csv", index=False
    )
    pd.DataFrame(test_metrics).to_csv(
        output_dir / "held_out_test_metrics.csv", index=False
    )
    pd.DataFrame(test_per_class).to_csv(
        output_dir / "held_out_test_per_class.csv", index=False
    )
    pd.DataFrame(predictions).to_csv(
        output_dir / "held_out_test_predictions.csv", index=False
    )
    best_name = str(cv_frame.iloc[0]["Model"])
    joblib.dump(fitted_models[best_name], output_dir / "best_model.joblib")
    save_run_summary(
        data,
        labels,
        args.input,
        train_indices,
        test_indices,
        args.seed,
        args.test_size,
        {
            "workflow": "traditional_ml",
            "folds": args.folds,
            "best_model_by_cv_macro_f1": best_name,
            "best_cv_macro_f1": float(cv_frame.iloc[0]["Macro F1"]),
        },
        output_dir,
    )
    print(f"\nRows: {len(data):,}; train: {len(train_y):,}; test: {len(test_y):,}")
    print("Cross-validation metrics:")
    print(cv_frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nHeld-out test metrics:")
    print(pd.DataFrame(test_metrics).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
