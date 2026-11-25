"""Experimental GNN using report-text similarity within Eclipse components."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F

from workflow_utils import (
    CATEGORICAL_COLUMNS,
    DEFAULT_INPUT,
    FEATURE_COLUMNS,
    TEXT_COLUMNS,
    create_output_dir,
    load_dataset,
    save_dataset_profile,
    save_run_summary,
    save_split,
    split_dataset,
)

DEFAULT_GNN_OUTPUT = Path(__file__).with_name("gnn_results")


class MeanGraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        sources, targets = edges
        neighbor_sums = torch.zeros_like(features)
        if sources.numel():
            neighbor_sums.index_add_(0, targets, features[sources])
        degrees = torch.bincount(targets, minlength=len(features)).to(features.dtype)
        neighbor_means = neighbor_sums / degrees.clamp_min(1).unsqueeze(1)
        return self.linear(torch.cat((features, neighbor_means), dim=1))


class BugReportGNN(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.first = MeanGraphConvolution(input_dim, hidden_dim)
        self.second = MeanGraphConvolution(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.first(features, edges))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.relu(self.second(hidden, edges))
        return self.output(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an experimental text/component graph neural network."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GNN_OUTPUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--text-components", type=int, default=96)
    parser.add_argument("--metadata-features", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if min(
        args.epochs,
        args.neighbors,
        args.max_features,
        args.text_components,
        args.metadata_features,
        args.hidden_dim,
    ) < 1:
        parser.error("epochs, neighbors, feature sizes, and hidden size must be positive")
    return args


def make_text(data: pd.DataFrame) -> pd.Series:
    return data[TEXT_COLUMNS[0]] + " " + data[TEXT_COLUMNS[1]]


def make_graph_edges(
    train_tfidf: Any,
    test_tfidf: Any,
    train_components: np.ndarray,
    test_components: np.ndarray,
    neighbors_per_node: int,
) -> torch.Tensor:
    train_groups: dict[str, list[int]] = {}
    for row, component in enumerate(train_components):
        train_groups.setdefault(str(component), []).append(row)

    edges: set[tuple[int, int]] = set()
    for rows in train_groups.values():
        if len(rows) < 2:
            continue
        neighbor_count = min(neighbors_per_node + 1, len(rows))
        search = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
        search.fit(train_tfidf[rows])
        nearest = search.kneighbors(train_tfidf[rows], return_distance=False)
        for local_index, candidates in enumerate(nearest):
            source = rows[local_index]
            for candidate in candidates:
                target = rows[int(candidate)]
                if source != target:
                    edges.add((source, target))
                    edges.add((target, source))

    test_groups: dict[str, list[int]] = {}
    for row, component in enumerate(test_components):
        category = str(component)
        if category in train_groups:
            test_groups.setdefault(category, []).append(row)
    train_count = len(train_components)
    for component, test_rows in test_groups.items():
        candidates = train_groups[component]
        neighbor_count = min(neighbors_per_node, len(candidates))
        search = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
        search.fit(train_tfidf[candidates])
        nearest = search.kneighbors(test_tfidf[test_rows], return_distance=False)
        for local_index, candidate_indices in enumerate(nearest):
            test_node = train_count + test_rows[local_index]
            for candidate_index in candidate_indices:
                train_node = candidates[int(candidate_index)]
                edges.add((train_node, test_node))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*edges)
    return torch.tensor([sources, targets], dtype=torch.long)


def main() -> None:
    args = parse_args()
    data, labels = load_dataset(args.input)
    train_indices, test_indices = split_dataset(data, args.test_size, args.seed)
    train_text = make_text(data.loc[train_indices])
    test_text = make_text(data.loc[test_indices])
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_ids = torch.tensor(
        [label_to_id[label] for label in data.loc[train_indices, "Priority"]],
        dtype=torch.long,
    )
    test_labels = data.loc[test_indices, "Priority"].tolist()
    test_ids = np.asarray([label_to_id[label] for label in test_labels], dtype=np.int64)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    train_tfidf = vectorizer.fit_transform(train_text)
    test_tfidf = vectorizer.transform(test_text)
    component_count = min(
        args.text_components,
        train_tfidf.shape[1] - 1,
        train_tfidf.shape[0] - 1,
    )
    if component_count < 1:
        raise ValueError("Not enough training text features for truncated SVD.")
    svd = TruncatedSVD(n_components=component_count, random_state=args.seed)
    train_text_features = svd.fit_transform(train_tfidf).astype(np.float32)
    test_text_features = svd.transform(test_tfidf).astype(np.float32)

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
    metadata_features = hasher.transform(category_dicts).toarray().astype(np.float32)
    train_metadata = metadata_features[data.index.get_indexer(train_indices)]
    test_metadata = metadata_features[data.index.get_indexer(test_indices)]
    train_features = np.concatenate((train_text_features, train_metadata), axis=1)
    test_features = np.concatenate((test_text_features, test_metadata), axis=1)
    all_features = np.concatenate((train_features, test_features), axis=0)

    edges = make_graph_edges(
        train_tfidf,
        test_tfidf,
        data.loc[train_indices, "Component"].to_numpy(),
        data.loc[test_indices, "Component"].to_numpy(),
        args.neighbors,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)
    edges = edges.to(device)
    train_ids_device = train_ids.to(device)
    class_counts = torch.bincount(train_ids, minlength=len(labels))
    class_weights = (
        len(train_ids) / (len(labels) * class_counts.clamp_min(1))
    ).float().to(device)

    model = BugReportGNN(
        all_features.shape[1], args.hidden_dim, len(labels), args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    epoch_losses: list[float] = []
    print(
        f"Training GNN on {device}; rows: {len(data):,}; "
        f"train: {len(train_indices):,}; test: {len(test_indices):,}; "
        f"directed edges: {edges.shape[1]:,}"
    )
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features_tensor, edges)
        loss = loss_fn(logits[: len(train_indices)], train_ids_device)
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"  Epoch {epoch + 1}/{args.epochs} - loss: {loss.item():.4f}")

    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(
            model(features_tensor, edges)[len(train_indices) :], dim=1
        ).cpu().numpy()
    predicted_ids = probabilities.argmax(axis=1)
    predicted_labels = [labels[int(index)] for index in predicted_ids]
    report = classification_report(
        test_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "Model": "GNN (text/component similarity graph)",
        "Accuracy": accuracy_score(test_labels, predicted_labels),
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1": report["weighted avg"]["f1-score"],
    }
    per_class = [
        {
            "Priority": label,
            "Precision": report[label]["precision"],
            "Recall": report[label]["recall"],
            "F1": report[label]["f1-score"],
            "Support": report[label]["support"],
        }
        for label in labels
    ]

    output_dir = create_output_dir(args.output_dir)
    save_dataset_profile(data, output_dir)
    save_split(data, train_indices, test_indices, output_dir)
    pd.DataFrame([metrics]).to_csv(
        output_dir / "held_out_test_metrics.csv", index=False
    )
    pd.DataFrame(per_class).to_csv(
        output_dir / "held_out_test_per_class.csv", index=False
    )
    pd.DataFrame(
        confusion_matrix(test_labels, predicted_labels, labels=labels),
        index=labels,
        columns=labels,
    ).rename_axis("Actual\\Predicted").to_csv(
        output_dir / "held_out_test_confusion_matrix.csv"
    )
    pd.DataFrame(
        {
            "Bug_ID": data.loc[test_indices, "Bug_ID"].to_numpy(),
            "Actual Priority": test_labels,
            "Predicted Priority": predicted_labels,
        }
    ).to_csv(output_dir / "held_out_test_predictions.csv", index=False)
    pd.DataFrame(
        {"Epoch": range(1, args.epochs + 1), "Loss": epoch_losses}
    ).to_csv(output_dir / "training_loss.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "input_dim": all_features.shape[1],
            "hidden_dim": args.hidden_dim,
            "num_classes": len(labels),
            "dropout": args.dropout,
            "priority_labels": labels,
            "graph_definition": (
                "cosine-nearest TF-IDF neighbors within Component; "
                "train edges connect training nodes only; test nodes receive "
                "messages only from training nodes"
            ),
        },
        output_dir / "gnn_model.pt",
    )
    joblib.dump(
        {"tfidf_vectorizer": vectorizer, "svd": svd, "feature_hasher": hasher},
        output_dir / "gnn_preprocessors.joblib",
    )
    save_run_summary(
        data,
        labels,
        args.input,
        train_indices,
        test_indices,
        args.seed,
        args.test_size,
        {
            "workflow": "experimental_gnn",
            "device": str(device),
            "epochs": args.epochs,
            "directed_graph_edges": int(edges.shape[1]),
            "graph_construction": (
                "TF-IDF cosine nearest neighbors within Component; test nodes "
                "connect only to training nodes"
            ),
            "graph_is_assumption": True,
            "feature_columns": FEATURE_COLUMNS,
        },
        output_dir,
    )
    print("\nHeld-out test metrics:")
    print(pd.DataFrame([metrics]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
