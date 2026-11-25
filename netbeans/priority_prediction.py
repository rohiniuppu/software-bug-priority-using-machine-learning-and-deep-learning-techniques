"""Compare classifiers for predicting NetBeans bug-report priority."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


DEFAULT_INPUT = Path(__file__).with_name("merged_bug_reports.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("priority_model_results")
TEXT_COLUMNS = ["Summary", "Keywords"]
CATEGORICAL_COLUMNS = ["Product", "Component", "OS", "Severity"]
FEATURE_COLUMNS = TEXT_COLUMNS + CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = {"Bug ID", "Priority", *FEATURE_COLUMNS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate models for predicting bug-report priority."
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


def create_output_dir(requested: Path) -> Path:
    if not requested.exists():
        requested.mkdir(parents=True)
        return requested

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = requested.with_name(f"{requested.name}_{timestamp}")
    suffix = 2
    while candidate.exists():
        candidate = requested.with_name(f"{requested.name}_{timestamp}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def make_pipeline(classifier: object) -> Pipeline:
    features = ColumnTransformer(
        [
            (
                "summary",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
                "Summary",
            ),
            (
                "keywords",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
                "Keywords",
            ),
            (
                "metadata",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
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
                n_estimators=200,
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


def get_report_rows(
    model_name: str,
    labels: list[str],
    actual: pd.Series,
    predicted: np.ndarray,
    split: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
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
    if not args.input.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.input}")

    original = pd.read_csv(args.input)
    missing_columns = REQUIRED_COLUMNS - set(original.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    output_dir = create_output_dir(args.output_dir)
    profile = pd.DataFrame(
        {
            "Column": original.columns,
            "Data Type": [str(dtype) for dtype in original.dtypes],
            "Missing Count": original.isna().sum().to_numpy(),
            "Missing Percent": (original.isna().mean() * 100).to_numpy(),
            "Unique Values": original.nunique(dropna=True).to_numpy(),
        }
    )
    profile.to_csv(output_dir / "dataset_profile.csv", index=False)
    overview = pd.DataFrame(
        [
            {"Measure": "Rows", "Value": len(original)},
            {"Measure": "Columns", "Value": len(original.columns)},
            {"Measure": "Full-row duplicates", "Value": int(original.duplicated().sum())},
            {
                "Measure": "Duplicate Bug IDs",
                "Value": int(original["Bug ID"].duplicated().sum()),
            },
            {"Measure": "Target", "Value": "Priority"},
        ]
    )
    overview.to_csv(output_dir / "dataset_overview.csv", index=False)

    data = original.copy()
    data["Priority"] = data["Priority"].astype("string").str.strip()
    data = data.loc[data["Priority"].notna() & data["Priority"].ne("")].copy()
    for column in TEXT_COLUMNS:
        data[column] = data[column].fillna("").astype(str).str.strip()
    for column in CATEGORICAL_COLUMNS:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    class_counts = data["Priority"].value_counts().sort_index()
    labels = sorted(class_counts.index.tolist())
    if len(labels) < 2:
        raise ValueError("Priority must contain at least two classes.")

    distribution = class_counts.rename_axis("Priority").reset_index(name="Count")
    distribution["Percent"] = distribution["Count"] / len(data) * 100
    distribution.to_csv(output_dir / "priority_distribution.csv", index=False)

    train_indices, test_indices = train_test_split(
        data.index,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data["Priority"],
    )
    train_y = data.loc[train_indices, "Priority"]
    test_y = data.loc[test_indices, "Priority"]
    if train_y.value_counts().min() < args.folds:
        raise ValueError(
            "Each training priority class must have at least as many rows "
            "as the requested number of CV folds."
        )
    train_x = data.loc[train_indices, FEATURE_COLUMNS]
    test_x = data.loc[test_indices, FEATURE_COLUMNS]

    pd.DataFrame(
        {
            "Bug ID": data.loc[train_indices.union(test_indices), "Bug ID"],
            "Split": [
                "train" if index in train_indices else "test"
                for index in train_indices.union(test_indices)
            ],
        }
    ).to_csv(output_dir / "split_assignments.csv", index=False)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    models = make_models(labels, args.seed)
    cv_metrics: list[dict[str, object]] = []
    cv_per_class: list[dict[str, object]] = []
    test_metrics: list[dict[str, object]] = []
    test_per_class: list[dict[str, object]] = []
    predictions: dict[str, object] = {
        "Bug ID": data.loc[test_indices, "Bug ID"].to_numpy(),
        "Actual Priority": test_y.to_numpy(),
    }
    fitted_models: dict[str, Pipeline] = {}

    for model_name, model in models.items():
        cv_prediction = cross_val_predict(
            model, train_x, train_y, cv=cv, method="predict", n_jobs=1
        )
        cv_summary, cv_classes = get_report_rows(
            model_name, labels, train_y, cv_prediction, "cross-validation"
        )
        cv_metrics.append(cv_summary)
        cv_per_class.extend(cv_classes)

        model.fit(train_x, train_y)
        test_prediction = model.predict(test_x)
        test_summary, test_classes = get_report_rows(
            model_name, labels, test_y, test_prediction, "held-out test"
        )
        test_metrics.append(test_summary)
        test_per_class.extend(test_classes)
        predictions[f"{model_name} Prediction"] = test_prediction
        matrix = confusion_matrix(test_y, test_prediction, labels=labels)
        pd.DataFrame(matrix, index=labels, columns=labels).rename_axis(
            "Actual\\Predicted"
        ).to_csv(output_dir / f"confusion_matrix_{model_name}.csv")
        fitted_models[model_name] = model

    cv_frame = pd.DataFrame(cv_metrics).sort_values(
        ["Macro F1", "Weighted F1"], ascending=False
    )
    cv_frame.to_csv(output_dir / "cross_validation_metrics.csv", index=False)
    pd.DataFrame(cv_per_class).to_csv(
        output_dir / "cross_validation_per_class.csv", index=False
    )
    pd.DataFrame(test_metrics).to_csv(output_dir / "held_out_test_metrics.csv", index=False)
    pd.DataFrame(test_per_class).to_csv(
        output_dir / "held_out_test_per_class.csv", index=False
    )
    pd.DataFrame(predictions).to_csv(output_dir / "held_out_test_predictions.csv", index=False)

    best_model_name = str(cv_frame.iloc[0]["Model"])
    joblib.dump(fitted_models[best_model_name], output_dir / "best_model.joblib")
    run_summary = {
        "input": str(args.input.resolve()),
        "target": "Priority",
        "rows_used": len(data),
        "train_rows": len(train_y),
        "test_rows": len(test_y),
        "class_labels": labels,
        "seed": args.seed,
        "folds": args.folds,
        "test_size": args.test_size,
        "features": FEATURE_COLUMNS,
        "excluded_columns": [
            column
            for column in original.columns
            if column not in FEATURE_COLUMNS and column != "Priority"
        ],
        "best_model_by_cv_macro_f1": best_model_name,
        "best_cv_macro_f1": float(cv_frame.iloc[0]["Macro F1"]),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    print(f"Dataset: {args.input}")
    print(
        f"Rows: {len(original):,}; used: {len(data):,}; "
        f"train: {len(train_y):,}; test: {len(test_y):,}"
    )
    print(f"Priority counts: {class_counts.to_dict()}")
    print(f"Best model by CV macro F1: {best_model_name}")
    print("\nCross-validation metrics:")
    print(cv_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nHeld-out test metrics:")
    print(
        pd.DataFrame(test_metrics)
        .set_index("Model")
        .loc[cv_frame["Model"]]
        .to_string(float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
