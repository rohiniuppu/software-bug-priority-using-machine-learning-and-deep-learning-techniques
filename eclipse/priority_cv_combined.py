"""Run per-fold and held-out priority metrics for Eclipse models."""

import argparse
import json
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
from sklearn.model_selection import StratifiedKFold

import deep_learning_prediction as dl
import gnn_prediction as gnn
from priority_prediction import FEATURE_COLUMNS, make_models
from workflow_utils import (
    CATEGORICAL_COLUMNS,
    DEFAULT_INPUT,
    TEXT_COLUMNS,
    load_dataset,
    split_dataset,
)


METRICS = ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1"]
MODEL_TITLES = {
    "majority_baseline": "Majority Baseline",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "linear_svm": "SVM",
    "naive_bayes": "Naive Bayes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five-fold and held-out metrics for Eclipse priority models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("priority_cv_results"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recurrent-epochs", type=int, default=6)
    parser.add_argument("--gnn-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=64)
    parser.add_argument("--recurrent-dropout", type=float, default=0.3)
    parser.add_argument("--gnn-dropout", type=float, default=0.4)
    parser.add_argument("--recurrent-learning-rate", type=float, default=0.001)
    parser.add_argument("--gnn-learning-rate", type=float, default=0.003)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--text-components", type=int, default=96)
    parser.add_argument("--metadata-features", type=int, default=64)
    args = parser.parse_args()
    if args.folds != 5:
        parser.error("This combined report requires exactly 5 folds.")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    if min(
        args.recurrent_epochs,
        args.gnn_epochs,
        args.batch_size,
        args.max_vocab_size,
        args.max_length,
        args.embedding_dim,
        args.recurrent_hidden_dim,
        args.hidden_dim,
        args.neighbors,
        args.max_features,
        args.text_components,
        args.metadata_features,
    ) < 1:
        parser.error("Epoch counts and model sizes must be positive.")
    return args


def calculate_metrics(
    actual: list[str], predicted: list[str], labels: list[str]
) -> dict[str, float]:
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"],
    }


def train_recurrent_split(
    model_type: str,
    data: pd.DataFrame,
    train_indices: pd.Index,
    valid_indices: pd.Index,
    labels: list[str],
    args: argparse.Namespace,
    fold: int,
) -> dict[str, float]:
    seed = args.seed + fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_text = dl.join_text(data.loc[train_indices])
    valid_text = dl.join_text(data.loc[valid_indices])
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_ids = torch.tensor(
        [label_to_id[label] for label in data.loc[train_indices, "Priority"]],
        dtype=torch.long,
    )
    valid_labels = data.loc[valid_indices, "Priority"].tolist()
    vocabulary = dl.build_vocabulary(train_text, args.max_vocab_size)
    train_tokens, train_lengths = dl.encode(
        train_text, vocabulary, args.max_length
    )
    valid_tokens, valid_lengths = dl.encode(
        valid_text, vocabulary, args.max_length
    )
    valid_ids = torch.tensor(
        [label_to_id[label] for label in valid_labels], dtype=torch.long
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
        valid_ids,
        args.batch_size,
        False,
        seed,
    )
    counts = torch.bincount(train_ids, minlength=len(labels))
    weights = len(train_ids) / (len(labels) * counts.clamp_min(1)).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dl.RecurrentClassifier(
        len(vocabulary),
        len(labels),
        model_type,
        args.embedding_dim,
        args.recurrent_hidden_dim,
        args.recurrent_dropout,
    )
    train_args = argparse.Namespace(
        epochs=args.recurrent_epochs,
        learning_rate=args.recurrent_learning_rate,
    )
    dl.train(model, train_loader, weights, device, train_args)
    probabilities = dl.predict(model, valid_loader, device)
    predicted = [labels[int(index)] for index in probabilities.argmax(axis=1)]
    return calculate_metrics(valid_labels, predicted, labels)


def train_gnn_split(
    data: pd.DataFrame,
    train_indices: pd.Index,
    valid_indices: pd.Index,
    labels: list[str],
    args: argparse.Namespace,
    fold: int,
) -> dict[str, float]:
    seed = args.seed + fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_text = gnn.make_text(data.loc[train_indices])
    valid_text = gnn.make_text(data.loc[valid_indices])
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=args.max_features,
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

    category_frame = data[CATEGORICAL_COLUMNS].astype(str)
    category_dicts = [
        {f"{column}={row[column]}": 1 for column in CATEGORICAL_COLUMNS}
        for _, row in category_frame.iterrows()
    ]
    hasher = FeatureHasher(
        n_features=args.metadata_features,
        input_type="dict",
        alternate_sign=False,
    )
    all_metadata = hasher.transform(category_dicts).toarray().astype(np.float32)
    ordered_positions = np.concatenate(
        (
            data.index.get_indexer(train_indices),
            data.index.get_indexer(valid_indices),
        )
    )
    features = np.concatenate(
        (
            np.concatenate((train_text_features, valid_text_features), axis=0),
            all_metadata[ordered_positions],
        ),
        axis=1,
    )
    edges = gnn.make_graph_edges(
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
    valid_labels = data.loc[valid_indices, "Priority"].tolist()
    class_counts = torch.bincount(train_ids, minlength=len(labels))
    weights = len(train_ids) / (len(labels) * class_counts.clamp_min(1)).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    edges = edges.to(device)
    model = gnn.BugReportGNN(
        features.shape[1], args.hidden_dim, len(labels), args.gnn_dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.gnn_learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    train_ids = train_ids.to(device)
    for epoch in range(args.gnn_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features_tensor, edges)
        loss = loss_fn(logits[: len(train_indices)], train_ids)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.gnn_epochs:
            print(
                f"    Epoch {epoch + 1}/{args.gnn_epochs} - loss: {loss.item():.4f}",
                flush=True,
            )
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(
            model(features_tensor, edges)[len(train_indices) :], dim=1
        ).cpu().numpy()
    predicted = [labels[int(index)] for index in probabilities.argmax(axis=1)]
    return calculate_metrics(valid_labels, predicted, labels)


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
                if len(match) != 1:
                    raise ValueError(f"Expected one result for {model}, fold {fold}.")
                row[model] = float(match.iloc[0][metric])
            rows.append(row)
    test_values = pd.concat([baseline_test, deep_test], ignore_index=True)
    for metric in METRICS:
        row = {"Measure": metric, "Evaluation": "Held-out test (20%)"}
        for model in models:
            match = test_values[test_values["Model"] == model]
            if len(match) != 1:
                raise ValueError(f"Expected one held-out result for {model}.")
            row[model] = float(match.iloc[0][metric])
        rows.append(row)
    return pd.DataFrame(rows, columns=["Measure", "Evaluation"] + models)


def main() -> None:
    args = parse_args()
    data, labels = load_dataset(args.input)
    train_indices, test_indices = split_dataset(data, args.test_size, args.seed)
    train_y = data.loc[train_indices, "Priority"]
    if train_y.value_counts().min() < args.folds:
        raise ValueError("Each class needs at least five training examples.")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_splits = list(splitter.split(train_indices, train_y))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {args.input.resolve()}", flush=True)
    print(
        f"Rows: {len(data):,}; train: {len(train_indices):,}; "
        f"test: {len(test_indices):,}",
        flush=True,
    )
    print(f"Priority counts: {data['Priority'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"Priority labels: {labels}", flush=True)
    print(
        "GNN metadata columns:",
        ", ".join(CATEGORICAL_COLUMNS),
        "(present)",
        flush=True,
    )
    print(
        "GNN graph note: existing implementation connects same-component "
        "TF-IDF neighbors; this is a graph assumption, not an explicit bug relation.",
        flush=True,
    )

    train_x = data.loc[train_indices, FEATURE_COLUMNS]
    test_x = data.loc[test_indices, FEATURE_COLUMNS]
    train_y = data.loc[train_indices, "Priority"]
    test_y = data.loc[test_indices, "Priority"]
    baseline_fold_rows: list[dict[str, Any]] = []
    baseline_test_rows: list[dict[str, Any]] = []
    for key, prototype in make_models(labels, args.seed).items():
        title = MODEL_TITLES[key]
        print(f"\n{title}: 5-fold CV", flush=True)
        for fold, (local_train, local_valid) in enumerate(fold_splits, start=1):
            model = clone(prototype)
            model.fit(train_x.iloc[local_train], train_y.iloc[local_train])
            predicted = model.predict(train_x.iloc[local_valid]).tolist()
            actual = train_y.iloc[local_valid].tolist()
            metrics = calculate_metrics(actual, predicted, labels)
            baseline_fold_rows.append({"Model": title, "Fold": fold, **metrics})
            pd.DataFrame(baseline_fold_rows).to_csv(
                args.output_dir / "baseline_cv_fold_metrics.csv", index=False
            )
            print(
                f"  Fold {fold}: "
                + ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items()),
                flush=True,
            )
        fitted = clone(prototype).fit(train_x, train_y)
        test_prediction = fitted.predict(test_x).tolist()
        baseline_test_rows.append(
            {
                "Model": title,
                **calculate_metrics(test_y.tolist(), test_prediction, labels),
            }
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
                + ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items()),
                flush=True,
            )

    print("\nGNN: 5-fold CV", flush=True)
    for fold, (local_train, local_valid) in enumerate(fold_splits, start=1):
        metrics = train_gnn_split(
            data,
            train_indices[local_train],
            train_indices[local_valid],
            labels,
            args,
            fold,
        )
        deep_fold_rows.append({"Model": "GNN", "Fold": fold, **metrics})
        pd.DataFrame(deep_fold_rows).to_csv(
            args.output_dir / "deep_learning_cv_fold_metrics.csv", index=False
        )
        print(
            f"  Fold {fold}: "
            + ", ".join(f"{name}: {value:.4f}" for name, value in metrics.items()),
            flush=True,
        )

    deep_test_rows: list[dict[str, Any]] = []
    for model_type in ["lstm", "rnn"]:
        metrics = train_recurrent_split(
            model_type, data, train_indices, test_indices, labels, args, 0
        )
        deep_test_rows.append({"Model": model_type.upper(), **metrics})
        pd.DataFrame(deep_test_rows).to_csv(
            args.output_dir / "deep_learning_held_out_test_metrics.csv", index=False
        )
    deep_test_rows.append(
        {
            "Model": "GNN",
            **train_gnn_split(data, train_indices, test_indices, labels, args, 0),
        }
    )
    deep_test = pd.DataFrame(deep_test_rows)
    baseline_folds = pd.read_csv(args.output_dir / "baseline_cv_fold_metrics.csv")
    deep_folds = pd.read_csv(args.output_dir / "deep_learning_cv_fold_metrics.csv")
    baseline_test = pd.read_csv(
        args.output_dir / "baseline_held_out_test_metrics.csv"
    )
    deep_test.to_csv(
        args.output_dir / "deep_learning_held_out_test_metrics.csv", index=False
    )
    combined = make_combined_table(
        baseline_folds, deep_folds, baseline_test, deep_test
    )
    combined.to_csv(args.output_dir / "combined_cv_and_dl_metrics.csv", index=False)
    gnn.save_split(data, train_indices, test_indices, args.output_dir)
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "rows": len(data),
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "labels": labels,
                "seed": args.seed,
                "folds": args.folds,
                "test_size": args.test_size,
                "features": FEATURE_COLUMNS,
                "recurrent_epochs": args.recurrent_epochs,
                "gnn_epochs": args.gnn_epochs,
                "gnn_metadata": CATEGORICAL_COLUMNS,
                "gnn_graph_is_assumption": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nCombined metrics (fold CV plus separate held-out test):")
    print(combined.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
