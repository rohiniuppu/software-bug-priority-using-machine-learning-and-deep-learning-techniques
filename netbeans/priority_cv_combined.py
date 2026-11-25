"""Run per-fold and held-out priority metrics for NetBeans models."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

import deep_learning_prediction as dl
from priority_prediction import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    make_models,
)


METRICS = ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]
MODEL_TITLES = {
    "majority_baseline": "Majority Baseline",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "linear_svm": "SVM",
    "naive_bayes": "Naive Bayes",
}
DL_MODELS = ["LSTM", "RNN", "GNN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five-fold and held-out metrics for NetBeans priority models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("merged_bug_reports.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("priority_cv_results"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--text-components", type=int, default=96)
    parser.add_argument("--metadata-features", type=int, default=64)
    args = parser.parse_args()
    if args.folds != 5:
        parser.error("This combined report requires exactly 5 folds.")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    if min(
        args.epochs,
        args.batch_size,
        args.max_vocab_size,
        args.max_length,
        args.embedding_dim,
        args.hidden_dim,
        args.neighbors,
        args.text_components,
        args.metadata_features,
    ) < 1:
        parser.error("Model sizes, epochs, batch size, and neighbors must be positive.")
    return args


def calculate_metrics(
    true_ids: np.ndarray, predicted_ids: np.ndarray, labels: list[str]
) -> dict[str, float]:
    true_labels = [labels[int(index)] for index in true_ids]
    predicted_labels = [labels[int(index)] for index in predicted_ids]
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"],
    }


def text_for_indices(data: pd.DataFrame, indices: np.ndarray) -> pd.Series:
    return (
        data.loc[indices, "Summary"].astype(str)
        + " "
        + data.loc[indices, "Component"].astype(str)
    )


def train_recurrent_split(
    model_type: str,
    data: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    labels: list[str],
    args: argparse.Namespace,
    fold: int,
) -> dict[str, float]:
    seed = args.seed + fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_text = text_for_indices(data, train_indices)
    valid_text = text_for_indices(data, valid_indices)
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_ids = torch.tensor(
        [label_to_id[label] for label in data.loc[train_indices, "Priority"]],
        dtype=torch.long,
    )
    valid_ids = np.asarray(
        [label_to_id[label] for label in data.loc[valid_indices, "Priority"]],
        dtype=np.int64,
    )
    vocabulary = dl.build_vocabulary(train_text, args.max_vocab_size)
    train_tokens, train_lengths = dl.encode_texts(
        train_text, vocabulary, args.max_length
    )
    valid_tokens, valid_lengths = dl.encode_texts(
        valid_text, vocabulary, args.max_length
    )
    train_loader = dl.make_loader(
        train_tokens,
        train_lengths,
        train_ids,
        args.batch_size,
        True,
        seed,
    )
    valid_loader = dl.make_loader(
        valid_tokens,
        valid_lengths,
        torch.tensor(valid_ids, dtype=torch.long),
        args.batch_size,
        False,
        seed,
    )
    class_weights = dl.class_weight_tensor(train_ids, len(labels))
    device = torch.device("cpu")
    model = dl.RecurrentClassifier(
        len(vocabulary),
        len(labels),
        model_type,
        args.embedding_dim,
        args.hidden_dim,
        args.dropout,
    )
    train_args = argparse.Namespace(epochs=args.epochs, learning_rate=args.learning_rate)
    dl.train_recurrent(model, train_loader, class_weights, device, train_args)
    probabilities = dl.predict_recurrent(model, valid_loader, device)
    return calculate_metrics(valid_ids, probabilities.argmax(axis=1), labels)


def train_gnn_split(
    data: pd.DataFrame,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    labels: list[str],
    args: argparse.Namespace,
    fold: int,
) -> dict[str, float]:
    seed = args.seed + fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_text = text_for_indices(data, train_indices)
    valid_text = text_for_indices(data, valid_indices)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    train_tfidf = vectorizer.fit_transform(train_text)
    valid_tfidf = vectorizer.transform(valid_text)
    component_count = min(
        args.text_components, train_tfidf.shape[1] - 1, train_tfidf.shape[0] - 1
    )
    if component_count < 1:
        raise ValueError(f"Fold {fold}: insufficient text features for truncated SVD.")
    svd = TruncatedSVD(n_components=component_count, random_state=seed)
    train_text_features = svd.fit_transform(train_tfidf).astype(np.float32)
    valid_text_features = svd.transform(valid_tfidf).astype(np.float32)

    categories = data[["Component"]].fillna("Unknown").astype(str)
    category_dicts = [
        {f"Component={row['Component']}": 1} for _, row in categories.iterrows()
    ]
    hasher = FeatureHasher(
        n_features=args.metadata_features,
        input_type="dict",
        alternate_sign=False,
    )
    metadata = hasher.transform(category_dicts).toarray().astype(np.float32)
    ordered_indices = np.concatenate((train_indices, valid_indices))
    features = np.concatenate(
        (
            np.concatenate((train_text_features, valid_text_features), axis=0),
            metadata[ordered_indices],
        ),
        axis=1,
    )
    edges = dl.build_graph_edges(
        train_tfidf,
        valid_tfidf,
        data.loc[train_indices, "Component"].to_numpy(),
        data.loc[valid_indices, "Component"].to_numpy(),
        args.neighbors,
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_ids = torch.tensor(
        [label_to_id[label] for label in data.loc[train_indices, "Priority"]],
        dtype=torch.long,
    )
    valid_ids = np.asarray(
        [label_to_id[label] for label in data.loc[valid_indices, "Priority"]],
        dtype=np.int64,
    )
    class_weights = dl.class_weight_tensor(train_ids, len(labels))
    device = torch.device("cpu")
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    edges = edges.to(device)
    model = dl.BugReportGNN(
        features.shape[1], args.hidden_dim, len(labels), args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    train_ids_device = train_ids.to(device)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features_tensor, edges)
        loss = criterion(logits[: len(train_indices)], train_ids_device)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(
                f"    Epoch {epoch + 1}/{args.epochs} - loss: {loss.item():.4f}",
                flush=True,
            )
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(
            model(features_tensor, edges)[len(train_indices) :], dim=1
        ).cpu().numpy()
    return calculate_metrics(valid_ids, probabilities.argmax(axis=1), labels)


def make_combined_table(
    baseline_folds: pd.DataFrame,
    deep_folds: pd.DataFrame,
    baseline_test: pd.DataFrame,
    deep_test: pd.DataFrame,
) -> pd.DataFrame:
    models = [
        "Majority Baseline",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "Naive Bayes",
        "LSTM",
        "RNN",
        "GNN",
    ]
    rows: list[dict[str, Any]] = []
    fold_values = pd.concat([baseline_folds, deep_folds], ignore_index=True)
    for metric in METRICS:
        for fold in range(1, 6):
            row: dict[str, Any] = {
                "Measure": metric,
                "Evaluation": f"Fold {fold} (5-fold CV)",
            }
            for model in models:
                match = fold_values[
                    (fold_values["Model"] == model)
                    & (fold_values["Fold"] == fold)
                ]
                row[model] = float(match.iloc[0][metric])
            rows.append(row)
    test_values = pd.concat([baseline_test, deep_test], ignore_index=True)
    for metric in METRICS:
        row = {"Measure": metric, "Evaluation": "Held-out test (20%)"}
        for model in models:
            match = test_values[test_values["Model"] == model]
            row[model] = float(match.iloc[0][metric])
        rows.append(row)
    return pd.DataFrame(rows, columns=["Measure", "Evaluation"] + models)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.input}")
    source = pd.read_csv(args.input)
    missing_columns = REQUIRED_COLUMNS - set(source.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    data = source.copy()
    data["Priority"] = data["Priority"].astype("string").str.strip()
    data = data.loc[data["Priority"].notna() & data["Priority"].ne("")].reset_index(
        drop=True
    )
    for column in dl.TEXT_COLUMNS:
        data[column] = data[column].fillna("").astype(str).str.strip()
    for column in dl.CATEGORY_COLUMNS:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    labels = sorted(data["Priority"].unique().tolist())
    if len(labels) < 2:
        raise ValueError("Priority must contain at least two classes.")
    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data["Priority"],
    )
    train_y = data.loc[train_indices, "Priority"]
    if train_y.value_counts().min() < args.folds:
        raise ValueError("Each training priority class must have at least five rows.")
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_splits = list(cv.split(train_indices, train_y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {args.input.resolve()}", flush=True)
    print(
        f"Rows: {len(source):,}; used: {len(data):,}; "
        f"train: {len(train_indices):,}; test: {len(test_indices):,}",
        flush=True,
    )
    print(f"Priority counts: {data['Priority'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"Priority labels: {labels}", flush=True)

    baseline_fold_rows: list[dict[str, Any]] = []
    baseline_test_rows: list[dict[str, Any]] = []
    prototypes = make_models(labels, args.seed)
    train_x = data.loc[train_indices, FEATURE_COLUMNS]
    train_y = data.loc[train_indices, "Priority"]
    test_x = data.loc[test_indices, FEATURE_COLUMNS]
    test_y = data.loc[test_indices, "Priority"]
    for key, prototype in prototypes.items():
        title = MODEL_TITLES[key]
        print(f"\n{title}: 5-fold CV", flush=True)
        for fold, (local_train, local_valid) in enumerate(fold_splits, start=1):
            estimator = clone(prototype)
            estimator.fit(train_x.iloc[local_train], train_y.iloc[local_train])
            prediction = estimator.predict(train_x.iloc[local_valid])
            label_to_id = {label: index for index, label in enumerate(labels)}
            true_ids = np.asarray(
                [label_to_id[label] for label in train_y.iloc[local_valid]]
            )
            predicted_ids = np.asarray([label_to_id[label] for label in prediction])
            metrics = calculate_metrics(true_ids, predicted_ids, labels)
            baseline_fold_rows.append({"Model": title, "Fold": fold, **metrics})
            pd.DataFrame(baseline_fold_rows).to_csv(
                args.output_dir / "baseline_cv_fold_metrics.csv", index=False
            )
            print(
                f"  Fold {fold}: "
                + ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items()),
                flush=True,
            )
        fitted = clone(prototype).fit(train_x, train_y)
        test_prediction = fitted.predict(test_x)
        label_to_id = {label: index for index, label in enumerate(labels)}
        true_ids = np.asarray([label_to_id[label] for label in test_y])
        predicted_ids = np.asarray([label_to_id[label] for label in test_prediction])
        baseline_test_rows.append(
            {"Model": title, **calculate_metrics(true_ids, predicted_ids, labels)}
        )
        pd.DataFrame(baseline_test_rows).to_csv(
            args.output_dir / "baseline_held_out_test_metrics.csv", index=False
        )

    deep_fold_rows: list[dict[str, Any]] = []
    for model_type in ["lstm", "rnn"]:
        title = model_type.upper()
        print(f"\n{title}: 5-fold CV", flush=True)
        for fold, (local_train, local_valid) in enumerate(fold_splits, start=1):
            fold_train = train_indices[local_train]
            fold_valid = train_indices[local_valid]
            metrics = train_recurrent_split(
                model_type, data, fold_train, fold_valid, labels, args, fold
            )
            deep_fold_rows.append({"Model": title, "Fold": fold, **metrics})
            pd.DataFrame(deep_fold_rows).to_csv(
                args.output_dir / "deep_learning_cv_fold_metrics.csv", index=False
            )
            print(
                f"  Fold {fold}: "
                + ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items()),
                flush=True,
            )

    print("\nGNN: 5-fold CV", flush=True)
    for fold, (local_train, local_valid) in enumerate(fold_splits, start=1):
        fold_train = train_indices[local_train]
        fold_valid = train_indices[local_valid]
        metrics = train_gnn_split(
            data, fold_train, fold_valid, labels, args, fold
        )
        deep_fold_rows.append({"Model": "GNN", "Fold": fold, **metrics})
        pd.DataFrame(deep_fold_rows).to_csv(
            args.output_dir / "deep_learning_cv_fold_metrics.csv", index=False
        )
        print(
            f"  Fold {fold}: "
            + ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items()),
            flush=True,
        )

    deep_test_rows: list[dict[str, Any]] = []
    for model_type in ["lstm", "rnn"]:
        metrics = train_recurrent_split(
            model_type, data, train_indices, test_indices, labels, args, 0
        )
        deep_test_rows.append(
            {"Model": model_type.upper(), **metrics}
        )
        pd.DataFrame(deep_test_rows).to_csv(
            args.output_dir / "deep_learning_held_out_test_metrics.csv", index=False
        )
    deep_test_rows.append(
        {
            "Model": "GNN",
            **train_gnn_split(
                data, train_indices, test_indices, labels, args, 0
            ),
        }
    )
    pd.DataFrame(deep_test_rows).to_csv(
        args.output_dir / "deep_learning_held_out_test_metrics.csv", index=False
    )

    baseline_folds = pd.DataFrame(baseline_fold_rows)
    deep_folds = pd.DataFrame(deep_fold_rows)
    baseline_test = pd.DataFrame(baseline_test_rows)
    deep_test = pd.DataFrame(deep_test_rows)
    baseline_folds.to_csv(args.output_dir / "baseline_cv_fold_metrics.csv", index=False)
    deep_folds.to_csv(args.output_dir / "deep_learning_cv_fold_metrics.csv", index=False)
    baseline_test.to_csv(args.output_dir / "baseline_held_out_test_metrics.csv", index=False)
    deep_test.to_csv(args.output_dir / "deep_learning_held_out_test_metrics.csv", index=False)

    combined = make_combined_table(
        baseline_folds, deep_folds, baseline_test, deep_test
    )
    combined.to_csv(args.output_dir / "combined_cv_and_dl_metrics.csv", index=False)
    training_set = set(train_indices)
    split_rows = pd.DataFrame(
        {
            "Bug ID": data["Bug ID"],
            "Split": [
                "train" if index in training_set else "test"
                for index in range(len(data))
            ],
        }
    )
    split_rows.to_csv(args.output_dir / "split_assignments.csv", index=False)
    (args.output_dir / "run_summary.json").write_text(
        pd.Series(
            {
                "input": str(args.input.resolve()),
                "rows": len(data),
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "labels": labels,
                "seed": args.seed,
                "folds": args.folds,
                "test_size": args.test_size,
                "feature_preset": "summary_component",
                "dl_epochs": args.epochs,
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )
    print("\nCombined metrics (fold CV plus separate held-out test):")
    print(combined.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
