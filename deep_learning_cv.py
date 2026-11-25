import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dl_utils import (
    RecurrentTextClassifier,
    build_vocabulary,
    encode_texts,
    make_loader,
    predict_probabilities,
    train_model,
)
from gnn_model import (
    BugReportGNN,
    build_test_edges,
    build_training_edges,
    edge_tensor,
    make_category_groups,
)
from model_utils import PRIORITY_LABELS, add_common_arguments, load_and_split


def calculate_metrics(true_ids: np.ndarray, predicted_ids: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_ids,
        predicted_ids,
        labels=np.arange(len(PRIORITY_LABELS)),
        average="weighted",
        zero_division=0,
    )
    return {
        "Accuracy": accuracy_score(true_ids, predicted_ids),
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Weighted F1": f1,
    }


def train_recurrent_fold(
    model_type: str,
    train_x: pd.Series,
    valid_x: pd.Series,
    train_y: pd.Series,
    valid_y: pd.Series,
    args: argparse.Namespace,
    fold: int,
) -> dict[str, float]:
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    label_to_id = {label: index for index, label in enumerate(PRIORITY_LABELS)}
    train_ids = torch.tensor([label_to_id[label] for label in train_y], dtype=torch.long)
    valid_ids = torch.tensor([label_to_id[label] for label in valid_y], dtype=torch.long)

    vocabulary = build_vocabulary(train_x, args.max_vocab_size)
    train_tokens, train_lengths = encode_texts(train_x, vocabulary, args.max_length)
    valid_tokens, valid_lengths = encode_texts(valid_x, vocabulary, args.max_length)
    class_counts = np.bincount(train_ids.numpy(), minlength=len(PRIORITY_LABELS))
    class_weights = torch.tensor(
        len(train_ids) / (len(PRIORITY_LABELS) * class_counts), dtype=torch.float32
    )
    train_loader = make_loader(
        train_tokens, train_lengths, train_ids, args.batch_size, True, args.seed + fold
    )
    valid_loader = make_loader(
        valid_tokens, valid_lengths, valid_ids, args.batch_size, False, args.seed + fold
    )
    device = torch.device("cpu")
    model = RecurrentTextClassifier(
        len(vocabulary),
        args.embedding_dim,
        args.hidden_dim,
        model_type,
        args.recurrent_dropout,
    )
    train_model(
        model,
        train_loader,
        class_weights,
        device,
        args.recurrent_epochs,
        args.recurrent_learning_rate,
    )
    actual_ids, probabilities = predict_probabilities(model, valid_loader, device)
    return calculate_metrics(actual_ids, probabilities.argmax(axis=1))


def train_gnn_fold(
    data: pd.DataFrame,
    train_y: pd.Series,
    train_indices: pd.Index,
    valid_indices: pd.Index,
    args: argparse.Namespace,
    fold: int,
) -> dict[str, float]:
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    train_positions = data.index.get_indexer(train_indices)
    valid_positions = data.index.get_indexer(valid_indices)
    train_text = data.loc[train_indices, "Summary"]
    valid_text = data.loc[valid_indices, "Summary"]

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
        raise ValueError(f"Fold {fold}: insufficient text features for truncated SVD")
    svd = TruncatedSVD(n_components=component_count, random_state=args.seed + fold)
    train_text_features = svd.fit_transform(train_tfidf).astype(np.float32)
    valid_text_features = svd.transform(valid_tfidf).astype(np.float32)

    category_columns = ["Product", "Component", "Classification", "OS"]
    missing_columns = set(category_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"GNN requires dataset columns: {sorted(missing_columns)}")
    category_values = data[category_columns].fillna("Unknown").astype(str)
    dictionaries = [
        {f"{column}={row[column]}": 1 for column in category_columns}
        for _, row in category_values.iterrows()
    ]
    hasher = FeatureHasher(
        n_features=args.metadata_features, input_type="dict", alternate_sign=False
    )
    metadata_features = hasher.transform(dictionaries).toarray().astype(np.float32)

    ordered_positions = np.concatenate((train_positions, valid_positions))
    features = np.concatenate(
        (
            np.concatenate((train_text_features, valid_text_features), axis=0),
            metadata_features[ordered_positions],
        ),
        axis=1,
    )
    train_count = len(train_positions)
    train_groups = make_category_groups(data, train_positions, "Component")
    train_edges = build_training_edges(
        train_tfidf, train_positions, train_groups, args.neighbors
    )
    valid_edges = build_test_edges(
        train_tfidf,
        valid_tfidf,
        train_positions,
        valid_positions,
        data,
        args.neighbors,
    )
    train_row_lookup = {
        int(position): row for row, position in enumerate(train_positions)
    }
    valid_row_lookup = {
        int(position): row + len(train_positions)
        for row, position in enumerate(valid_positions)
    }
    valid_edges = {
        (train_row_lookup[source], valid_row_lookup[target])
        for source, target in valid_edges
    }
    edges = edge_tensor(train_edges, valid_edges)

    label_to_id = {label: index for index, label in enumerate(PRIORITY_LABELS)}
    train_ids = torch.tensor([label_to_id[label] for label in train_y], dtype=torch.long)
    valid_y = data.loc[valid_indices, "Priority"]
    valid_ids = torch.tensor([label_to_id[label] for label in valid_y], dtype=torch.long)
    class_counts = np.bincount(train_ids.numpy(), minlength=len(PRIORITY_LABELS))
    class_weights = torch.tensor(
        len(train_ids) / (len(PRIORITY_LABELS) * class_counts), dtype=torch.float32
    )

    device = torch.device("cpu")
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    edges = edges.to(device)
    train_ids = train_ids.to(device)
    model = BugReportGNN(features.shape[1], args.hidden_dim, args.gnn_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.gnn_learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    for epoch in range(args.gnn_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features_tensor, edges)
        loss = criterion(logits[:train_count], train_ids)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.gnn_epochs:
            print(f"    Epoch {epoch + 1}/{args.gnn_epochs} - loss: {loss.item():.4f}", flush=True)

    model.eval()
    with torch.inference_mode():
        predicted_ids = model(features_tensor, edges)[train_count:].argmax(dim=1).cpu().numpy()
    return calculate_metrics(valid_ids.numpy(), predicted_ids)


def make_combined_table(
    baseline_folds: pd.DataFrame,
    deep_folds: pd.DataFrame,
    output_dir: Path,
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
    metrics = [
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
    ]
    fold_scores = pd.concat([baseline_folds, deep_folds], ignore_index=True)
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for fold in range(1, 6):
            row: dict[str, Any] = {
                "Measure": metric,
                "Evaluation": f"Fold {fold} (5-fold CV)",
            }
            for model in models:
                match = fold_scores[
                    (fold_scores["Model"] == model) & (fold_scores["Fold"] == fold)
                ]
                row[model] = float(match.iloc[0][metric]) if not match.empty else np.nan
            rows.append(row)

    test_metrics: dict[str, pd.Series] = {}
    for key in ["decision_tree", "random_forest", "svm", "naive_bayes"]:
        values = pd.read_csv(output_dir.parent / "baseline_results" / key / "held_out_test_metrics.csv")
        test_metrics[str(values.iloc[0]["Model"])] = values.iloc[0]
    for key in ["lstm", "rnn", "gnn"]:
        values = pd.read_csv(
            output_dir.parent / "deep_learning_results" / key / "held_out_test_metrics.csv"
        )
        test_metrics[str(values.iloc[0]["Model"])] = values.iloc[0]

    majority = pd.read_csv(
        output_dir.parent / "baseline_results" / "majority_baseline_per_class.csv"
    )
    total = majority["support"].sum()
    test_metrics["Majority Baseline"] = pd.Series(
        {
            "Accuracy": majority.loc[majority["Priority"] == "P3", "support"].iloc[0]
            / total,
            "Weighted Precision": (majority["precision"] * majority["support"]).sum()
            / total,
            "Weighted Recall": (majority["recall"] * majority["support"]).sum() / total,
            "Weighted F1": (majority["f1-score"] * majority["support"]).sum() / total,
        }
    )
    for metric in metrics:
        row = {"Measure": metric, "Evaluation": "Held-out test (20%)"}
        for model in models:
            row[model] = float(test_metrics[model][metric]) if model in test_metrics else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=["Measure", "Evaluation"] + models)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 5-fold CV for LSTM, RNN, and GNN and print combined metrics."
    )
    add_common_arguments(parser)
    parser.set_defaults(output_dir=Path(__file__).with_name("deep_learning_cv_results"))
    parser.add_argument("--recurrent-epochs", type=int, default=8)
    parser.add_argument("--gnn-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--recurrent-dropout", type=float, default=0.3)
    parser.add_argument("--gnn-dropout", type=float, default=0.4)
    parser.add_argument("--recurrent-learning-rate", type=float, default=0.001)
    parser.add_argument("--gnn-learning-rate", type=float, default=0.003)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--text-components", type=int, default=128)
    parser.add_argument("--metadata-features", type=int, default=64)
    args = parser.parse_args()
    if args.folds != 5:
        parser.error("This combined report requires exactly 5 folds.")
    if min(args.recurrent_epochs, args.gnn_epochs, args.batch_size, args.max_length) < 1:
        parser.error("Epoch counts, batch-size, and max-length must be positive.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data, train_x, _, train_y, _ = load_and_split(args)
    splitter = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=args.seed
    )
    fold_file = output_dir / "deep_learning_cv_fold_metrics.csv"
    if fold_file.exists():
        deep_rows = pd.read_csv(fold_file).to_dict("records")
    else:
        deep_rows = []
    completed = {(row["Model"], int(row["Fold"])) for row in deep_rows}

    for model_type in ["lstm", "rnn"]:
        model_title = model_type.upper()
        print(f"\n{model_title}: 5-fold CV", flush=True)
        for fold, (train_positions, valid_positions) in enumerate(
            splitter.split(train_x, train_y), start=1
        ):
            if (model_title, fold) in completed:
                print(f"  Fold {fold}/5 already saved; skipping.", flush=True)
                continue
            print(f"  Fold {fold}/5", flush=True)
            metrics = train_recurrent_fold(
                model_type,
                train_x.iloc[train_positions],
                train_x.iloc[valid_positions],
                train_y.iloc[train_positions],
                train_y.iloc[valid_positions],
                args,
                fold,
            )
            deep_rows.append({"Model": model_title, "Fold": fold, **metrics})
            pd.DataFrame(deep_rows).to_csv(fold_file, index=False)
            completed.add((model_title, fold))
            print("    " + ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items()), flush=True)

    print("\nGNN: 5-fold CV", flush=True)
    for fold, (train_positions, valid_positions) in enumerate(
        splitter.split(train_x, train_y), start=1
    ):
        if ("GNN", fold) in completed:
            print(f"  Fold {fold}/5 already saved; skipping.", flush=True)
            continue
        print(f"  Fold {fold}/5", flush=True)
        metrics = train_gnn_fold(
            data,
            train_y.iloc[train_positions],
            train_x.index[train_positions],
            train_x.index[valid_positions],
            args,
            fold,
        )
        deep_rows.append({"Model": "GNN", "Fold": fold, **metrics})
        pd.DataFrame(deep_rows).to_csv(fold_file, index=False)
        completed.add(("GNN", fold))
        print("    " + ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items()), flush=True)

    deep_folds = pd.DataFrame(deep_rows)
    baseline_folds = pd.read_csv(
        Path(__file__).with_name("baseline_results") / "training_cv_fold_metrics.csv"
    )
    combined = make_combined_table(baseline_folds, deep_folds, output_dir)
    combined.to_csv(output_dir / "combined_cv_and_dl_metrics.csv", index=False)
    combined.to_csv(
        Path(__file__).with_name("baseline_results")
        / "combined_cv_and_dl_metrics.csv",
        index=False,
    )
    print("\nCombined metrics (fold CV plus separate held-out test):")
    print(combined.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nFold metrics: {(output_dir / 'deep_learning_cv_fold_metrics.csv').resolve()}")
    print(f"Combined table: {(output_dir / 'combined_cv_and_dl_metrics.csv').resolve()}")
    print(
        "Baseline results copy: "
        f"{(Path(__file__).with_name('baseline_results') / 'combined_cv_and_dl_metrics.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
