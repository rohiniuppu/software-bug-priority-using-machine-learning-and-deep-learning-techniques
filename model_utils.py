import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


PRIORITY_LABELS = ["P1", "P2", "P3", "P4", "P5"]
MODEL_FACTORIES = {
    "decision_tree": lambda seed: DecisionTreeClassifier(
        class_weight="balanced", random_state=seed
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    ),
    "svm": lambda seed: LinearSVC(class_weight="balanced", random_state=seed),
    "naive_bayes": lambda seed: MultinomialNB(),
}
MODEL_TITLES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "naive_bayes": "Naive Bayes",
}


class FoldBalancedLinearSVC(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        C: float = 1.0,
        class_weight_factors: tuple[float, ...] = (1, 1, 1, 1, 1),
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.class_weight_factors = class_weight_factors
        self.random_state = random_state

    def fit(self, features: Any, labels: Any) -> "FoldBalancedLinearSVC":
        counts = pd.Series(labels).value_counts()
        base_weight = len(labels) / (len(counts) * counts)
        factors = dict(zip(PRIORITY_LABELS, self.class_weight_factors))
        weights = {
            label: float(base_weight[label] * factors[label])
            for label in counts.index
        }
        self.classifier_ = LinearSVC(
            C=self.C,
            class_weight=weights,
            max_iter=10000,
            random_state=self.random_state,
        ).fit(features, labels)
        self.classes_ = self.classifier_.classes_
        return self

    def predict(self, features: Any) -> Any:
        return self.classifier_.predict(features)

    def decision_function(self, features: Any) -> Any:
        return self.classifier_.decision_function(features)


def build_pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def build_augmented_pipeline(classifier: Any) -> Pipeline:
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
                ["Product", "Component", "Classification", "OS"],
            ),
        ]
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def get_augmented_features(data: pd.DataFrame, indices: pd.Index) -> pd.DataFrame:
    columns = [
        "Summary",
        "Keywords",
        "Product",
        "Component",
        "Classification",
        "OS",
    ]
    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Augmented SVM requires CSV columns: {sorted(missing_columns)}"
        )

    features = data.loc[indices, columns].copy()
    for column in ("Summary", "Keywords"):
        features[column] = features[column].fillna("").astype(str)
    for column in ("Product", "Component", "Classification", "OS"):
        features[column] = features[column].fillna("Unknown").astype(str)
    return features


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("bugs-2026-10-15.csv"),
        help="Path to the bug-report CSV.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("baseline_results"),
        help="Directory for generated result files.",
    )


def load_and_split(args: argparse.Namespace) -> tuple[pd.DataFrame, ...]:
    data = pd.read_csv(args.input)
    required_columns = {"Bug ID", "Summary", "Priority"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    data["Summary"] = data["Summary"].fillna("").astype(str).str.strip()
    data["Priority"] = data["Priority"].astype("string").str.strip()
    data = data.loc[data["Summary"].ne("") & data["Priority"].notna()].copy()

    unexpected_labels = sorted(set(data["Priority"].unique()) - set(PRIORITY_LABELS))
    if unexpected_labels:
        raise ValueError(f"Unexpected priority labels: {unexpected_labels}")
    if data["Priority"].value_counts().min() < args.folds:
        raise ValueError("Every priority class must have at least one row per CV fold.")

    train_x, test_x, train_y, test_y = train_test_split(
        data["Summary"],
        data["Priority"],
        test_size=args.test_size,
        stratify=data["Priority"],
        random_state=args.seed,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    priority_counts = (
        data["Priority"].value_counts().rename_axis("Priority").reset_index(name="Count")
    )
    priority_counts["Percent"] = priority_counts["Count"] / len(data) * 100
    priority_counts.to_csv(output_dir / "priority_distribution.csv", index=False)

    summary_lengths = data["Summary"].str.len()
    overview = pd.DataFrame(
        [
            {"Measure": "Rows in source", "Value": len(data)},
            {"Measure": "Missing summaries", "Value": data["Summary"].isna().sum()},
            {"Measure": "Missing priorities", "Value": data["Priority"].isna().sum()},
            {"Measure": "Duplicate Bug IDs", "Value": data["Bug ID"].duplicated().sum()},
            {"Measure": "Unique priorities", "Value": data["Priority"].nunique()},
            {"Measure": "Summary characters (min)", "Value": summary_lengths.min()},
            {"Measure": "Summary characters (median)", "Value": summary_lengths.median()},
            {"Measure": "Summary characters (max)", "Value": summary_lengths.max()},
            {"Measure": "Rows used for modeling", "Value": len(data)},
        ]
    )
    overview.to_csv(output_dir / "dataset_overview.csv", index=False)

    split_rows = []
    for split_name, indices, labels in (
        ("train", train_x.index, train_y),
        ("test", test_x.index, test_y),
    ):
        split_rows.append(
            pd.DataFrame(
                {
                    "Bug ID": data.loc[indices, "Bug ID"].to_numpy(),
                    "Priority": labels.to_numpy(),
                    "Split": split_name,
                }
            )
        )
    pd.concat(split_rows, ignore_index=True).to_csv(
        output_dir / "fixed_train_test_split.csv", index=False
    )
    return data, train_x, test_x, train_y, test_y


def evaluate_model(
    model_key: str,
    args: argparse.Namespace,
    output_dir: Path,
    tune_svm: bool = False,
    include_metadata: bool = False,
) -> dict[str, Any]:
    data, train_x, test_x, train_y, test_y = load_and_split(args)
    if include_metadata:
        train_x = get_augmented_features(data, train_x.index)
        test_x = get_augmented_features(data, test_x.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    if model_key == "svm" and tune_svm:
        pipeline_builder = build_augmented_pipeline if include_metadata else build_pipeline
        estimator = pipeline_builder(FoldBalancedLinearSVC(random_state=args.seed))
        parameter_grid = {
            "classifier__C": [0.03, 0.1, 0.3],
            "classifier__class_weight_factors": [
                (1, 1, 1, 1, 1),
                (8, 2, 1, 3, 1.5),
                (16, 2, 1, 4, 1.5),
                (32, 3, 1, 6, 2),
                (64, 4, 1, 8, 2),
            ],
            "tfidf__ngram_range": [(1, 1), (1, 2)],
        }
        inner_cv = StratifiedKFold(
            n_splits=max(3, args.folds - 2), shuffle=True, random_state=args.seed
        )
        estimator = GridSearchCV(
            estimator,
            parameter_grid,
            scoring={"macro_f1": "f1_macro", "macro_recall": "recall_macro"},
            refit="macro_f1",
            cv=inner_cv,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_predictions = cross_val_predict(
            estimator, train_x, train_y, cv=cv, n_jobs=1
        )
    else:
        classifier = (
            FoldBalancedLinearSVC(
                C=0.1,
                class_weight_factors=(64, 4, 1, 8, 2),
                random_state=args.seed,
            )
            if include_metadata
            else MODEL_FACTORIES[model_key](args.seed)
        )
        pipeline_builder = build_augmented_pipeline if include_metadata else build_pipeline
        estimator = pipeline_builder(classifier)
        cv_predictions = cross_val_predict(
            estimator, train_x, train_y, cv=cv, n_jobs=-1
        )
    cv_report = classification_report(
        train_y,
        cv_predictions,
        labels=PRIORITY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    cv_per_class = pd.DataFrame.from_dict(
        {label: cv_report[label] for label in PRIORITY_LABELS}, orient="index"
    )
    cv_per_class.index.name = "Priority"
    cv_per_class.to_csv(output_dir / "training_cv_per_class.csv")

    cv_metrics = pd.DataFrame(
        [
            {
                "Model": MODEL_TITLES[model_key],
                "Accuracy": accuracy_score(train_y, cv_predictions),
                "Macro Precision": cv_report["macro avg"]["precision"],
                "Macro Recall (MAR)": cv_report["macro avg"]["recall"],
                "Macro F1": cv_report["macro avg"]["f1-score"],
                "Weighted Precision": cv_report["weighted avg"]["precision"],
                "Weighted Recall": cv_report["weighted avg"]["recall"],
                "Weighted F1": cv_report["weighted avg"]["f1-score"],
            }
        ]
    )
    cv_metrics.to_csv(output_dir / "training_cv_metrics.csv", index=False)

    estimator.fit(train_x, train_y)
    if model_key == "svm" and tune_svm:
        pd.DataFrame(estimator.cv_results_).to_csv(
            output_dir / "svm_tuning_cv_results.csv", index=False
        )
        pd.DataFrame([{"Best parameters": str(estimator.best_params_)}]).to_csv(
            output_dir / "selected_svm_parameters.csv", index=False
        )
        fitted_model = estimator.best_estimator_
    else:
        fitted_model = estimator

    test_predictions = fitted_model.predict(test_x)
    test_report = classification_report(
        test_y,
        test_predictions,
        labels=PRIORITY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    if hasattr(fitted_model, "predict_proba"):
        class_scores = fitted_model.predict_proba(test_x)
    else:
        class_scores = fitted_model.decision_function(test_x)
    model_classes = list(fitted_model.classes_)
    class_scores = class_scores[:, [model_classes.index(label) for label in PRIORITY_LABELS]]
    binary_test_y = label_binarize(test_y, classes=PRIORITY_LABELS)

    test_metrics = pd.DataFrame(
        [
            {
                "Model": MODEL_TITLES[model_key],
                "Accuracy": accuracy_score(test_y, test_predictions),
                "Macro Precision": test_report["macro avg"]["precision"],
                "Macro Recall (MAR)": test_report["macro avg"]["recall"],
                "Macro F1": test_report["macro avg"]["f1-score"],
                "Weighted Precision": test_report["weighted avg"]["precision"],
                "Weighted Recall": test_report["weighted avg"]["recall"],
                "Weighted F1": test_report["weighted avg"]["f1-score"],
                "MAP (Macro Avg Precision, OVR)": average_precision_score(
                    binary_test_y, class_scores, average="macro"
                ),
            }
        ]
    )
    test_metrics.to_csv(output_dir / "held_out_test_metrics.csv", index=False)

    test_per_class = pd.DataFrame.from_dict(
        {label: test_report[label] for label in PRIORITY_LABELS}, orient="index"
    )
    test_per_class.index.name = "Priority"
    test_per_class.to_csv(output_dir / "held_out_test_per_class.csv")

    matrix = confusion_matrix(test_y, test_predictions, labels=PRIORITY_LABELS)
    pd.DataFrame(matrix, index=PRIORITY_LABELS, columns=PRIORITY_LABELS).rename_axis(
        "Actual\\Predicted"
    ).to_csv(output_dir / "held_out_test_confusion_matrix.csv")

    return {
        "model": MODEL_TITLES[model_key],
        "rows": len(data),
        "train_rows": len(train_y),
        "test_rows": len(test_y),
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "output_dir": output_dir,
    }


def run_single_model(model_key: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Evaluate {MODEL_TITLES[model_key]} for bug priority prediction."
    )
    add_common_arguments(parser)
    if model_key == "svm":
        svm_options = parser.add_mutually_exclusive_group()
        svm_options.add_argument(
            "--tune",
            action="store_true",
            help="Tune SVM class weights, C, and TF-IDF n-grams using nested CV.",
        )
        svm_options.add_argument(
            "--include-metadata",
            action="store_true",
            help="Use keywords and bug metadata with the summary for SVM.",
        )
    args = parser.parse_args()
    output_dir = args.output_dir / model_key
    tune_svm = model_key == "svm" and args.tune
    include_metadata = model_key == "svm" and args.include_metadata
    if tune_svm:
        output_dir = args.output_dir / "svm_tuned"
    elif include_metadata:
        output_dir = args.output_dir / "svm_augmented"
    results = evaluate_model(
        model_key,
        args,
        output_dir,
        tune_svm=tune_svm,
        include_metadata=include_metadata,
    )
    print(f"Model: {results['model']}")
    print(f"Rows: {results['rows']:,}; train: {results['train_rows']:,}; test: {results['test_rows']:,}")
    print("\nTraining cross-validation:")
    print(results["cv_metrics"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nHeld-out test metrics:")
    print(results["test_metrics"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Results written to: {results['output_dir'].resolve()}")


def run_all_models() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all four bug-priority classifiers with a majority baseline."
    )
    add_common_arguments(parser)
    args = parser.parse_args()
    all_results = []

    for model_key in MODEL_FACTORIES:
        all_results.append(
            evaluate_model(model_key, args, args.output_dir / model_key)
        )

    data, train_x, _, train_y, _ = load_and_split(args)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    predictions = cross_val_predict(
        DummyClassifier(strategy="most_frequent"), train_x, train_y, cv=cv, n_jobs=-1
    )
    dummy_report = classification_report(
        train_y,
        predictions,
        labels=PRIORITY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    all_rows = [
        {
            "Model": "Majority baseline",
            "Accuracy": accuracy_score(train_y, predictions),
            "Macro Precision": dummy_report["macro avg"]["precision"],
            "Macro Recall (MAR)": dummy_report["macro avg"]["recall"],
            "Macro F1": dummy_report["macro avg"]["f1-score"],
            "Weighted F1": dummy_report["weighted avg"]["f1-score"],
        }
    ]
    all_rows.extend(
        result["cv_metrics"].iloc[0].to_dict() for result in all_results
    )
    comparison = pd.DataFrame(all_rows)
    comparison.to_csv(args.output_dir / "training_cv_comparison.csv", index=False)
    print(f"Compared {len(MODEL_FACTORIES)} models on {len(data):,} rows.")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Individual model outputs: {args.output_dir.resolve()}")