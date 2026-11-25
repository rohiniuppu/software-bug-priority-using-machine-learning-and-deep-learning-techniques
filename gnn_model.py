import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F

from dl_utils import save_evaluation
from model_utils import PRIORITY_LABELS, add_common_arguments, load_and_split


class MeanGraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        sources, targets = edge_index
        neighbor_sums = torch.zeros_like(features)
        neighbor_sums.index_add_(0, targets, features[sources])
        degrees = torch.bincount(targets, minlength=len(features)).to(features.dtype)
        neighbor_means = neighbor_sums / degrees.clamp_min(1).unsqueeze(1)
        return self.linear(torch.cat((features, neighbor_means), dim=1))


class BugReportGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.first = MeanGraphConvolution(input_dim, hidden_dim)
        self.second = MeanGraphConvolution(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.output = nn.Linear(hidden_dim, len(PRIORITY_LABELS))

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.first(features, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.relu(self.second(hidden, edge_index))
        return self.output(hidden)


def make_category_groups(
    data: pd.DataFrame, positions: np.ndarray, column: str
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    values = data[column].fillna("Unknown").astype(str).to_numpy()
    for position in positions:
        groups[values[position]].append(int(position))
    return groups


def build_training_edges(
    train_tfidf: Any,
    train_positions: np.ndarray,
    component_groups: dict[str, list[int]],
    neighbors_per_node: int,
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    train_row_lookup = {
        int(position): row for row, position in enumerate(train_positions)
    }
    for group_positions in component_groups.values():
        if len(group_positions) < 2:
            continue
        group_array = np.asarray(group_positions, dtype=np.int64)
        row_indices = np.asarray(
            [train_row_lookup[int(position)] for position in group_array],
            dtype=np.int64,
        )
        group_features = train_tfidf[row_indices]
        neighbor_count = min(neighbors_per_node + 1, len(group_positions))
        search = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
        search.fit(group_features)
        nearest = search.kneighbors(group_features, return_distance=False)
        for local_index, candidates in enumerate(nearest):
            source_position = train_row_lookup[int(group_array[local_index])]
            for candidate_index in candidates:
                neighbor_position = train_row_lookup[
                    int(group_array[candidate_index])
                ]
                if neighbor_position == source_position:
                    continue
                edges.add((source_position, neighbor_position))
                edges.add((neighbor_position, source_position))
    return edges


def build_test_edges(
    train_tfidf: Any,
    test_tfidf: Any,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    data: pd.DataFrame,
    neighbors_per_node: int,
) -> set[tuple[int, int]]:
    component_groups = make_category_groups(data, train_positions, "Component")
    product_groups = make_category_groups(data, train_positions, "Product")
    component_values = data["Component"].fillna("Unknown").astype(str).to_numpy()
    product_values = data["Product"].fillna("Unknown").astype(str).to_numpy()
    group_queries: dict[tuple[str, str], list[int]] = defaultdict(list)
    for test_position in test_positions:
        component = component_values[test_position]
        product = product_values[test_position]
        if component_groups.get(component):
            group_queries[("Component", component)].append(int(test_position))
        elif product_groups.get(product):
            group_queries[("Product", product)].append(int(test_position))

    edges: set[tuple[int, int]] = set()
    test_row_lookup = {
        int(position): row + len(train_positions)
        for row, position in enumerate(test_positions)
    }
    test_sparse_lookup = {
        int(position): row for row, position in enumerate(test_positions)
    }
    train_row_lookup = {
        int(position): row for row, position in enumerate(train_positions)
    }
    for (field, category), query_positions in group_queries.items():
        candidate_positions = (
            component_groups[category]
            if field == "Component"
            else product_groups[category]
        )
        candidate_rows = np.asarray(
            [train_row_lookup[position] for position in candidate_positions],
            dtype=np.int64,
        )
        neighbor_count = min(neighbors_per_node, len(candidate_positions))
        search = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
        search.fit(train_tfidf[candidate_rows])
        query_rows = np.asarray(
            [test_sparse_lookup[position] for position in query_positions], dtype=np.int64
        )
        nearest = search.kneighbors(test_tfidf[query_rows], return_distance=False)
        for query_index, candidates in enumerate(nearest):
            target_position = int(query_positions[query_index])
            for candidate_index in candidates:
                source_position = int(candidate_positions[candidate_index])
                edges.add((source_position, target_position))
    return edges


def edge_tensor(
    train_edges: set[tuple[int, int]],
    test_edges: set[tuple[int, int]],
) -> torch.Tensor:
    combined = train_edges | test_edges
    if not combined:
        return torch.empty((2, 0), dtype=torch.long)
    sources, targets = zip(*combined)
    return torch.tensor([sources, targets], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a graph neural network for bug priority prediction."
    )
    add_common_arguments(parser)
    parser.set_defaults(output_dir=Path(__file__).with_name("deep_learning_results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--text-components", type=int, default=128)
    parser.add_argument("--metadata-features", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()
    if min(args.epochs, args.neighbors, args.text_components, args.metadata_features) < 1:
        parser.error("epochs, neighbors, text-components, and metadata-features must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    data, train_x, test_x, train_y, test_y = load_and_split(
        argparse.Namespace(
            input=args.input,
            folds=args.folds,
            test_size=args.test_size,
            seed=args.seed,
            output_dir=output_dir,
        )
    )
    baseline_split = Path(__file__).with_name("baseline_results") / "fixed_train_test_split.csv"
    saved_split = output_dir / "fixed_train_test_split.csv"
    if baseline_split.exists():
        expected = pd.read_csv(baseline_split).sort_values("Bug ID").reset_index(drop=True)
        actual = pd.read_csv(saved_split).sort_values("Bug ID").reset_index(drop=True)
        if not expected.equals(actual):
            raise ValueError("Generated split does not match baseline_results/fixed_train_test_split.csv")

    required = {"Component", "Product", "Classification", "OS"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"GNN requires dataset columns: {sorted(missing)}")
    train_positions = data.index.get_indexer(train_x.index)
    test_positions = data.index.get_indexer(test_x.index)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    train_tfidf = vectorizer.fit_transform(train_x)
    test_tfidf = vectorizer.transform(test_x)
    component_count = min(
        args.text_components, train_tfidf.shape[1] - 1, train_tfidf.shape[0] - 1
    )
    if component_count < 1:
        raise ValueError("Not enough training text features for truncated SVD")
    svd = TruncatedSVD(n_components=component_count, random_state=args.seed)
    train_text_features = svd.fit_transform(train_tfidf).astype(np.float32)
    test_text_features = svd.transform(test_tfidf).astype(np.float32)

    category_columns = ["Product", "Component", "Classification", "OS"]
    category_values = data[category_columns].fillna("Unknown").astype(str)
    dictionaries = [
        {f"{column}={row[column]}": 1 for column in category_columns}
        for _, row in category_values.iterrows()
    ]
    hasher = FeatureHasher(
        n_features=args.metadata_features, input_type="dict", alternate_sign=False
    )
    metadata_features = hasher.transform(dictionaries).toarray().astype(np.float32)
    ordered_positions = np.concatenate((train_positions, test_positions))
    text_features = np.concatenate((train_text_features, test_text_features), axis=0)
    all_features = np.concatenate(
        (text_features, metadata_features[ordered_positions]), axis=1
    )
    train_count = len(train_positions)
    training_component_groups = make_category_groups(data, train_positions, "Component")
    train_edges = build_training_edges(
        train_tfidf, train_positions, training_component_groups, args.neighbors
    )
    test_edges = build_test_edges(
        train_tfidf,
        test_tfidf,
        train_positions,
        test_positions,
        data,
        args.neighbors,
    )
    edges = edge_tensor(train_edges, test_edges)

    label_to_id = {label: index for index, label in enumerate(PRIORITY_LABELS)}
    train_ids = torch.tensor([label_to_id[label] for label in train_y], dtype=torch.long)
    test_ids = torch.tensor([label_to_id[label] for label in test_y], dtype=torch.long)
    class_counts = np.bincount(train_ids.numpy(), minlength=len(PRIORITY_LABELS))
    class_weights = torch.tensor(
        len(train_ids) / (len(PRIORITY_LABELS) * class_counts), dtype=torch.float32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)
    edges = edges.to(device)
    train_ids = train_ids.to(device)
    model = BugReportGNN(all_features.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print(f"Model: GNN | device: {device}")
    print(f"Rows: {len(data):,}; train: {len(train_y):,}; test: {len(test_y):,}")
    print(f"Graph nodes: {len(data):,}; directed message-passing edges: {edges.shape[1]:,}")
    print("Training class counts:", dict(zip(PRIORITY_LABELS, class_counts.tolist())))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features_tensor, edges)
        loss = criterion(logits[:train_count], train_ids)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"Epoch {epoch + 1}/{args.epochs} - loss: {loss.item():.4f}")

    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(features_tensor, edges)[train_count:], dim=1)
    metrics = save_evaluation(
        "gnn",
        output_dir,
        test_ids.cpu().numpy(),
        probabilities.cpu().numpy(),
        test_x.index,
        data,
    )
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "input_dim": all_features.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "priority_labels": PRIORITY_LABELS,
            "graph_definition": "same-component TF-IDF neighbors; test nodes only receive train-node edges",
        },
        output_dir / "gnn_model.pt",
    )
    joblib.dump(
        {"tfidf_vectorizer": vectorizer, "svd": svd},
        output_dir / "gnn_text_preprocessors.joblib",
    )
    config: dict[str, Any] = vars(args).copy()
    config.update(
        {
            "model_type": "gnn",
            "device": str(device),
            "actual_text_components": component_count,
            "graph_edges": int(edges.shape[1]),
            "graph_definition": "same-component TF-IDF neighbors; test nodes only receive train-node edges",
        }
    )
    (output_dir / "model_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print("\nHeld-out test metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()